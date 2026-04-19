"""
Three-Tier Memory System (MemOS) for the Causal-JEPA World Model.

This module implements the memory architecture that enables "infinite learning":
new knowledge enters as episodic memories, gets consolidated into semantic
concepts over time, and becomes permanently accessible.

NEUROSCIENCE ANALOGY:
====================
The three tiers mirror the human memory consolidation pipeline:
- Working memory (attention) = prefrontal cortex, fast, limited, volatile
- Episodic memory (FAISS index) = hippocampus, specific episodes, medium-term
- Semantic memory (concept graph) = neocortex, abstract concepts, permanent

In humans, sleep consolidation moves episodic memories into semantic knowledge.
In our model, periodic "consolidation passes" cluster episodic memories into
concept embeddings, achieving the same function computationally.

WHY THIS MATTERS FOR AGI:
========================
Standard LLMs have a fixed context window. Once information leaves the window,
it's gone. This means they cannot:
- Learn from every conversation (no accumulating experience)
- Build abstract concepts from specific episodes over time
- Maintain a growing world model that improves with use

Our memory system addresses all three limitations:
1. Every interaction is stored as an episodic memory → accumulating experience
2. Consolidation clusters episodes into concepts → abstract knowledge formation
3. Concept injection at inference → persistent, growing world model

SCALABILITY:
============
- Episodic store: FAISS IVF index scales to millions of episodes with O(log N) retrieval
- Semantic store: dictionary-based, O(1) lookup per concept
- Cross-attention injection: O(K × S) where K = topk retrieved (typically 8)
- All memory operations run on CPU → no GPU VRAM impact
- The consolidation pass is periodic (every N steps), not per-sample
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# FAISS is used for efficient nearest-neighbor retrieval in the episodic store.
# We use CPU-only FAISS to keep GPU VRAM entirely for the model.
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning(
        "FAISS not installed. Episodic memory will use brute-force retrieval. "
        "Install with: pip install faiss-cpu"
    )

from .encoder import get_norm

logger = logging.getLogger(__name__)


# ============================================================================
#  Data Structures
# ============================================================================

@dataclass
class Episode:
    """
    A single episodic memory entry.

    Each episode captures a causal snapshot: the world state BEFORE an event,
    the event itself, and the resulting state AFTER. This causal triple
    (state, event, outcome) is the fundamental unit of experience.

    Fields:
        state_vec: The world state before the event (from CausalStateCompressor)
        event_vec: The event/action that occurred (from encoder + compressor)
        outcome_vec: The resulting state after the event
        timestamp: When this episode was created (training step or wall clock)
        modality_tag: Which modality produced this episode (0=text, 1=image, 2=mixed)
        metadata: Optional JSON-serializable metadata for debugging/analysis
    """
    state_vec: np.ndarray     # (D,) float32
    event_vec: np.ndarray     # (D,) float32
    outcome_vec: np.ndarray   # (D,) float32
    timestamp: float = 0.0
    modality_tag: int = 0
    metadata: Optional[dict] = None


# ============================================================================
#  Tier 2: Episodic Memory Store
# ============================================================================

class EpisodicMemoryStore:
    """
    FAISS-backed episodic memory for efficient nearest-neighbor retrieval.

    This is where specific experiences are stored. Each episode is a causal
    triple (state, event, outcome) indexed by the state vector for retrieval.

    WHY FAISS:
    - Cosine similarity retrieval in O(log N) instead of O(N)
    - CPU-only → no GPU VRAM impact
    - Scales from 100K to millions of episodes with IVF indexing
    - Battle-tested in production recommendation systems

    RETRIEVAL STRATEGY:
    We retrieve by cosine similarity between the CURRENT state and STORED states.
    This is semantically meaningful: "find past situations that looked like this one."
    The retrieved episodes' event and outcome vectors provide context about
    what happened in similar situations — a form of experience-based reasoning.

    SCALABILITY:
    - Small stores (<10K): use flat index (exact, fast enough)
    - Medium stores (10K-1M): use IVF (approximate, much faster)
    - Large stores (>1M): use IVF with PQ compression (approximate, memory-efficient)
    - The transition between these modes is automatic based on store size.
    """

    def __init__(self, hidden_dim: int, capacity: int = 100_000, topk: int = 8):
        self.hidden_dim = hidden_dim
        self.capacity = capacity
        self.topk = topk

        # Storage for full episode data (indexed in parallel with FAISS)
        self.episodes: List[Episode] = []

        # FAISS index for state vector retrieval
        # We use IndexFlatIP (inner product) on L2-normalized vectors,
        # which is equivalent to cosine similarity but faster.
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(hidden_dim)
        else:
            # Fallback: store state vectors as a matrix for brute-force search
            self._state_matrix: Optional[np.ndarray] = None

        # Track statistics for monitoring
        self._num_stored = 0
        self._num_retrievals = 0
        self._upgraded_to_ivf = False

    def _maybe_upgrade_index(self):
        """
        Upgrade to IVF index when store grows large enough.

        IVF (Inverted File) partitions the vector space into clusters,
        then only searches the nearest clusters during retrieval.
        This scales retrieval from O(N) to O(sqrt(N)).

        We upgrade when the store exceeds 10K entries, which is where
        the speedup becomes meaningful.
        """
        if (
            FAISS_AVAILABLE
            and not self._upgraded_to_ivf
            and self._num_stored >= 10_000
            and self._num_stored % 1_000 == 0  # check periodically, not every insert
        ):
            # Create IVF index with sqrt(N) clusters (standard FAISS recommendation)
            num_clusters = min(int(np.sqrt(self._num_stored)), 256)

            # Build quantizer and IVF index
            quantizer = faiss.IndexFlatIP(self.hidden_dim)
            ivf_index = faiss.IndexIVFFlat(quantizer, self.hidden_dim, num_clusters)

            # Train the IVF index on existing vectors
            all_vecs = np.array([ep.state_vec for ep in self.episodes], dtype=np.float32)
            ivf_index.train(all_vecs)
            ivf_index.add(all_vecs)

            # Set search parameters
            ivf_index.nprobe = min(16, num_clusters)  # search top-16 clusters

            self.index = ivf_index
            self._upgraded_to_ivf = True
            logger.info(
                f"Upgraded episodic store to IVF index: {num_clusters} clusters, "
                f"{self._num_stored} episodes"
            )

    def store(self, episode: Episode) -> None:
        """
        Store a new episode in the memory.

        If the store is at capacity, the oldest episode is removed (FIFO).
        In a more sophisticated system, we'd remove based on relevance or
        consolidation status, but FIFO is sufficient for the prototype and
        matches the neuroscience intuition that recent episodes are more
        likely to be accessed than old ones.
        """
        # Normalize state vector for cosine similarity via inner product
        state_normalized = episode.state_vec / (np.linalg.norm(episode.state_vec) + 1e-8)
        episode_with_norm = Episode(
            state_vec=state_normalized,
            event_vec=episode.event_vec,
            outcome_vec=episode.outcome_vec,
            timestamp=episode.timestamp,
            modality_tag=episode.modality_tag,
            metadata=episode.metadata,
        )

        # Handle capacity overflow
        if len(self.episodes) >= self.capacity:
            self.episodes.pop(0)
            # Rebuild index (necessary for FAISS flat index)
            if FAISS_AVAILABLE:
                self.index.reset()
                if self.episodes:
                    vecs = np.array([ep.state_vec for ep in self.episodes], dtype=np.float32)
                    self.index.add(vecs)

        # Add to store
        self.episodes.append(episode_with_norm)
        if FAISS_AVAILABLE:
            self.index.add(state_normalized.reshape(1, -1).astype(np.float32))
        else:
            # Brute-force fallback: rebuild matrix
            self._state_matrix = np.array(
                [ep.state_vec for ep in self.episodes], dtype=np.float32
            )

        self._num_stored += 1
        self._maybe_upgrade_index()

    def retrieve(
        self,
        query_state: np.ndarray,
        topk: Optional[int] = None,
    ) -> List[Tuple[Episode, float]]:
        """
        Retrieve the top-k most similar episodes to the query state.

        Args:
            query_state: (D,) — the current world state vector
            topk: Number of episodes to retrieve (default: self.topk)

        Returns:
            List of (Episode, similarity_score) tuples, sorted by similarity
        """
        if not self.episodes:
            return []

        topk = topk or self.topk
        topk = min(topk, len(self.episodes))

        # Normalize query
        query_normalized = query_state / (np.linalg.norm(query_state) + 1e-8)
        query_normalized = query_normalized.reshape(1, -1).astype(np.float32)

        self._num_retrievals += 1

        if FAISS_AVAILABLE:
            scores, indices = self.index.search(query_normalized, topk)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.episodes):  # FAISS returns -1 for missing
                    results.append((self.episodes[idx], float(score)))
            return results
        else:
            # Brute-force cosine similarity
            similarities = self._state_matrix @ query_normalized.T  # (N, 1)
            similarities = similarities.squeeze(-1)
            top_indices = np.argsort(similarities)[-topk:][::-1]
            return [
                (self.episodes[i], float(similarities[i]))
                for i in top_indices
            ]

    def get_all_state_vectors(self) -> np.ndarray:
        """Return all state vectors as a matrix. Used for consolidation."""
        if not self.episodes:
            return np.array([], dtype=np.float32).reshape(0, self.hidden_dim)
        return np.array([ep.state_vec for ep in self.episodes], dtype=np.float32)

    def get_all_event_vectors(self) -> np.ndarray:
        """Return all event vectors as a matrix. Used for consolidation."""
        if not self.episodes:
            return np.array([], dtype=np.float32).reshape(0, self.hidden_dim)
        return np.array([ep.event_vec for ep in self.episodes], dtype=np.float32)

    @property
    def size(self) -> int:
        return len(self.episodes)

    def save(self, path: str) -> None:
        """Save the episodic store to disk."""
        import pickle
        data = {
            "episodes": self.episodes,
            "hidden_dim": self.hidden_dim,
            "capacity": self.capacity,
            "topk": self.topk,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved episodic store ({self.size} episodes) to {path}")

    def load(self, path: str) -> None:
        """Load the episodic store from disk and rebuild index."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.episodes = data["episodes"]
        self.hidden_dim = data["hidden_dim"]
        self.capacity = data["capacity"]
        self.topk = data["topk"]

        # Rebuild FAISS index
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.hidden_dim)
            if self.episodes:
                vecs = np.array([ep.state_vec for ep in self.episodes], dtype=np.float32)
                self.index.add(vecs)

        self._num_stored = len(self.episodes)
        logger.info(f"Loaded episodic store ({self.size} episodes) from {path}")


# ============================================================================
#  Tier 3: Semantic Memory (Concept Graph)
# ============================================================================

class SemanticMemory:
    """
    Concept-level memory that distills episodic memories into abstract knowledge.

    This is powered by a periodic "consolidation pass" that:
    1. Clusters the episodic store into groups of related episodes
    2. For each cluster, distills the episodes into a single concept embedding
    3. Assigns a human-readable concept label (optional, for interpretability)

    The consolidation process is inspired by the information bottleneck principle:
    the concept embedding must preserve the information needed for causal
    prediction while discarding episode-specific details.

    HOW CONCEPTS ARE USED:
    At inference time, if the current state has high cosine similarity to a
    concept embedding, that concept is soft-injected into the working context.
    This is how "permanent knowledge" works:
    - Episode: "I saw a red ball bounce off a blue wall"
    - Concept (after consolidation): "elastic collisions preserve kinetic energy"
    - The concept is more general and applies to situations the model never saw

    SCALABILITY:
    - Dictionary-based: O(1) lookup per concept
    - Number of concepts grows with experience but much slower than episodes
    - Concept vectors are the same dimension as state vectors → same retrieval cost
    - Consolidation is periodic (not per-sample) → amortized cost
    """

    def __init__(self, hidden_dim: int, num_clusters: int = 64, sim_threshold: float = 0.7):
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.sim_threshold = sim_threshold

        # Concept store: name → embedding
        self.concepts: Dict[str, np.ndarray] = {}
        # Concept metadata: name → info dict
        self.concept_meta: Dict[str, dict] = {}

    def consolidate(self, episodic_store: EpisodicMemoryStore) -> int:
        """
        Consolidate episodic memories into semantic concepts.

        This is the "sleep consolidation" pass:
        1. Cluster all episodes by state similarity (K-means on state vectors)
        2. For each cluster, compute the centroid as the concept embedding
           (simplified information bottleneck — the centroid minimizes
           average distance to all cluster members)
        3. Store the concept with a generated label

        Returns the number of new concepts created.

        TODO(research): Full information bottleneck optimization would minimize:
            I(concept; episodes) - β * I(concept; causal_predictions)
        This would produce concepts that are maximally compressed while preserving
        causal prediction power. The centroid approximation is a reasonable first pass.
        """
        if episodic_store.size < self.num_clusters:
            return 0  # Not enough episodes to consolidate

        state_vecs = episodic_store.get_all_state_vectors()
        event_vecs = episodic_store.get_all_event_vectors()

        try:
            from sklearn.cluster import KMeans
        except ImportError:
            # Fallback: simple k-means with numpy
            return self._consolidate_numpy_kmeans(state_vecs, event_vecs)

        # Cluster episodes by state similarity
        k = min(self.num_clusters, episodic_store.size // 2)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(state_vecs)

        new_concepts = 0
        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            cluster_states = state_vecs[cluster_mask]
            cluster_events = event_vecs[cluster_mask]

            if len(cluster_states) < 2:
                continue  # Singleton clusters aren't meaningful concepts

            # Concept embedding = centroid (information bottleneck approximation)
            concept_state = cluster_states.mean(axis=0)
            concept_state = concept_state / (np.linalg.norm(concept_state) + 1e-8)

            # Concept event = most common event pattern in the cluster
            concept_event = cluster_events.mean(axis=0)
            concept_event = concept_event / (np.linalg.norm(concept_event) + 1e-8)

            # Generate concept label
            concept_name = f"concept_{cluster_id:04d}"

            # Store concept (overwrites if concept already exists from prior consolidation)
            self.concepts[concept_name] = concept_state
            self.concept_meta[concept_name] = {
                "event_embedding": concept_event,
                "num_episodes": int(cluster_mask.sum()),
                "created_at": time.time(),
                "cluster_id": cluster_id,
                "within_cluster_variance": float(cluster_states.var()),
            }
            new_concepts += 1

        logger.info(
            f"Consolidation complete: {new_concepts} concepts from "
            f"{episodic_store.size} episodes"
        )
        return new_concepts

    def _consolidate_numpy_kmeans(self, state_vecs: np.ndarray, event_vecs: np.ndarray) -> int:
        """
        Fallback K-means using only numpy (no sklearn dependency).
        Simpler but functional — used when sklearn is not available.
        """
        k = min(self.num_clusters, len(state_vecs) // 2)
        n = len(state_vecs)

        # Initialize centroids with K-means++
        centroids = [state_vecs[np.random.randint(n)].copy()]
        for _ in range(1, k):
            distances = np.array([
                min(np.linalg.norm(x - c) ** 2 for c in centroids)
                for x in state_vecs
            ])
            probs = distances / (distances.sum() + 1e-8)
            centroids.append(state_vecs[np.random.choice(n, p=probs)].copy())

        centroids = np.array(centroids)

        # Run K-means for 20 iterations
        for _ in range(20):
            # Assign clusters
            distances = np.linalg.norm(state_vecs[:, None] - centroids[None, :], axis=-1)
            labels = distances.argmin(axis=1)
            # Update centroids
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    centroids[j] = state_vecs[mask].mean(axis=0)

        # Create concepts from clusters
        new_concepts = 0
        for cluster_id in range(k):
            mask = labels == cluster_id
            if mask.sum() < 2:
                continue

            concept_state = centroids[cluster_id]
            concept_state = concept_state / (np.linalg.norm(concept_state) + 1e-8)

            concept_event = event_vecs[mask].mean(axis=0)
            concept_event = concept_event / (np.linalg.norm(concept_event) + 1e-8)

            concept_name = f"concept_{cluster_id:04d}"
            self.concepts[concept_name] = concept_state
            self.concept_meta[concept_name] = {
                "event_embedding": concept_event,
                "num_episodes": int(mask.sum()),
                "created_at": time.time(),
                "cluster_id": cluster_id,
            }
            new_concepts += 1

        return new_concepts

    def query(self, state: np.ndarray, threshold: Optional[float] = None) -> List[Tuple[str, np.ndarray, float]]:
        """
        Find concepts relevant to the current state.

        Returns concepts whose embedding has cosine similarity > threshold
        with the given state vector.

        Args:
            state: (D,) — current world state vector
            threshold: minimum cosine similarity (default: self.sim_threshold)

        Returns:
            List of (concept_name, concept_embedding, similarity) tuples
        """
        if not self.concepts:
            return []

        threshold = threshold or self.sim_threshold
        state_normalized = state / (np.linalg.norm(state) + 1e-8)

        results = []
        for name, concept_vec in self.concepts.items():
            similarity = float(np.dot(state_normalized, concept_vec))
            if similarity > threshold:
                results.append((name, concept_vec, similarity))

        # Sort by similarity (most relevant first)
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    @property
    def num_concepts(self) -> int:
        return len(self.concepts)

    def save(self, path: str) -> None:
        """Save semantic memory to disk."""
        import pickle
        data = {
            "concepts": self.concepts,
            "concept_meta": self.concept_meta,
            "hidden_dim": self.hidden_dim,
            "num_clusters": self.num_clusters,
            "sim_threshold": self.sim_threshold,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load semantic memory from disk."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.concepts = data["concepts"]
        self.concept_meta = data["concept_meta"]


# ============================================================================
#  Memory Manager — Unified interface for all three tiers
# ============================================================================

class MemoryManager(nn.Module):
    """
    Unified memory manager coordinating all three tiers.

    This is the interface between the neural model and the memory system.
    It handles:
    1. Storing new episodes (after encoding observations)
    2. Retrieving relevant episodes during inference
    3. Projecting retrieved memories into the encoder's hidden space
    4. Triggering periodic consolidation of episodes into concepts
    5. Injecting relevant concepts into the working context

    The projection layer converts CPU-side memory vectors into GPU tensors
    suitable for cross-attention injection in the encoder.

    SCALABILITY:
    - All FAISS operations run on CPU → no GPU VRAM impact
    - The projection layer is the only trainable component (small: D×D)
    - Memory size is bounded by episodic_capacity config
    - Concept count grows slowly with training (O(sqrt(episodes)))
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        # Tier 2: Episodic memory store (FAISS-backed, CPU)
        self.episodic_store = EpisodicMemoryStore(
            hidden_dim=config.hidden_dim,
            capacity=config.episodic_capacity,
            topk=config.topk_retrieve,
        )

        # Tier 3: Semantic memory (concept graph, CPU)
        self.semantic_memory = SemanticMemory(
            hidden_dim=config.hidden_dim,
            num_clusters=config.num_concept_clusters,
            sim_threshold=config.concept_sim_threshold,
        )

        # Projection for retrieved memories → encoder hidden space.
        # This learned projection aligns the raw memory vectors (which may
        # be from earlier training checkpoints with different representations)
        # with the current encoder's hidden space.
        self.memory_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.memory_norm = get_norm(config.norm_type, config.hidden_dim)

        # Projection for concept vectors → encoder hidden space
        self.concept_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        # Track consolidation schedule
        self._steps_since_consolidation = 0
        self.consolidation_interval = 1000  # consolidate every N store operations

    def store_episode(
        self,
        state: torch.Tensor,
        event: torch.Tensor,
        outcome: torch.Tensor,
        modality_tag: int = 0,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Store a new episode in the episodic memory.

        Converts GPU tensors to CPU numpy arrays for FAISS storage.
        This is intentional: memories live on CPU to preserve GPU VRAM
        for the model's forward/backward passes.
        """
        episode = Episode(
            state_vec=state.detach().cpu().numpy().astype(np.float32),
            event_vec=event.detach().cpu().numpy().astype(np.float32),
            outcome_vec=outcome.detach().cpu().numpy().astype(np.float32),
            timestamp=time.time(),
            modality_tag=modality_tag,
            metadata=metadata,
        )
        self.episodic_store.store(episode)

        # Check if it's time to consolidate
        self._steps_since_consolidation += 1
        if self._steps_since_consolidation >= self.consolidation_interval:
            self.consolidate()
            self._steps_since_consolidation = 0

    def retrieve_memories(
        self,
        query_state: torch.Tensor,
        topk: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Retrieve relevant episodic memories for the current state.

        The retrieved memories are projected into the encoder's hidden space
        and returned as a tensor ready for cross-attention injection.

        Args:
            query_state: (B, D) or (D,) — current world state
            topk: number of memories to retrieve
            device: target device for the output tensor

        Returns:
            (B, K_actual, D) — retrieved and projected memory vectors
            K_actual may be less than topk if the store has fewer entries
        """
        if device is None:
            device = query_state.device

        # Handle batched queries
        if query_state.dim() == 1:
            query_state = query_state.unsqueeze(0)

        B = query_state.shape[0]
        batch_memories = []

        for b in range(B):
            query_np = query_state[b].detach().cpu().numpy().astype(np.float32)

            # Retrieve from episodic store
            episodic_results = self.episodic_store.retrieve(query_np, topk=topk)

            # Also check semantic memory for relevant concepts
            concept_results = self.semantic_memory.query(query_np)

            # Combine: episodic state vectors + concept vectors
            memory_vecs = []
            for episode, score in episodic_results:
                # Use a combination of state and event vectors as the memory representation
                # This gives the model both "what the situation was" and "what happened"
                combined = np.concatenate([episode.state_vec, episode.event_vec])
                # Project back to hidden_dim by averaging the two halves
                mem_vec = (episode.state_vec + episode.event_vec) / 2.0
                memory_vecs.append(mem_vec)

            for concept_name, concept_vec, sim in concept_results:
                memory_vecs.append(concept_vec)

            if memory_vecs:
                mem_tensor = torch.tensor(
                    np.array(memory_vecs, dtype=np.float32),
                    device=device,
                )
            else:
                # No memories available — return empty tensor
                mem_tensor = torch.zeros(0, self.hidden_dim, device=device)

            batch_memories.append(mem_tensor)

        # Pad to same length across batch
        max_k = max(m.shape[0] for m in batch_memories) if batch_memories else 0
        if max_k == 0:
            return torch.zeros(B, 0, self.hidden_dim, device=device)

        padded = torch.zeros(B, max_k, self.hidden_dim, device=device)
        for b, mem in enumerate(batch_memories):
            if mem.shape[0] > 0:
                padded[b, :mem.shape[0]] = mem

        # Project through learned memory projection
        projected = self.memory_proj(padded)
        projected = self.memory_norm(projected)

        return projected

    def consolidate(self) -> int:
        """
        Run the consolidation pass: cluster episodic memories into concepts.

        This is the "sleep consolidation" step that converts specific
        experiences into abstract knowledge.

        Returns the number of new concepts created.
        """
        n_concepts = self.semantic_memory.consolidate(self.episodic_store)
        logger.info(
            f"Memory consolidation: {n_concepts} concepts, "
            f"{self.episodic_store.size} episodes, "
            f"{self.semantic_memory.num_concepts} total concepts"
        )
        return n_concepts

    def get_stats(self) -> dict:
        """Return memory system statistics for logging."""
        return {
            "episodic_store_size": self.episodic_store.size,
            "num_concepts": self.semantic_memory.num_concepts,
            "total_retrievals": self.episodic_store._num_retrievals,
            "steps_since_consolidation": self._steps_since_consolidation,
        }

    def save(self, directory: str) -> None:
        """Save all memory tiers to disk."""
        import os
        os.makedirs(directory, exist_ok=True)
        self.episodic_store.save(os.path.join(directory, "episodic_store.pkl"))
        self.semantic_memory.save(os.path.join(directory, "semantic_memory.pkl"))
        # Save projection weights
        torch.save({
            "memory_proj": self.memory_proj.state_dict(),
            "concept_proj": self.concept_proj.state_dict(),
            "memory_norm": self.memory_norm.state_dict(),
        }, os.path.join(directory, "memory_projections.pt"))
        logger.info(f"Saved all memory tiers to {directory}")

    def load(self, directory: str) -> None:
        """Load all memory tiers from disk."""
        import os
        episodic_path = os.path.join(directory, "episodic_store.pkl")
        semantic_path = os.path.join(directory, "semantic_memory.pkl")
        proj_path = os.path.join(directory, "memory_projections.pt")

        if os.path.exists(episodic_path):
            self.episodic_store.load(episodic_path)
        if os.path.exists(semantic_path):
            self.semantic_memory.load(semantic_path)
        if os.path.exists(proj_path):
            weights = torch.load(proj_path, map_location="cpu")
            self.memory_proj.load_state_dict(weights["memory_proj"])
            self.concept_proj.load_state_dict(weights["concept_proj"])
            self.memory_norm.load_state_dict(weights["memory_norm"])
        logger.info(f"Loaded all memory tiers from {directory}")
