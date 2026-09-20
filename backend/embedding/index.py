"""IRIS Vector Index — FAISS HNSW index management and persistence."""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

logger = logging.getLogger("iris.embedding.index")

DEFAULT_INDEX_PATH = Path("data/faiss_index.bin")
HNSW_EF_SEARCH = 64

_stores: Dict[Path, "FaissVectorStore"] = {}
_stores_lock = threading.Lock()


def get_vector_store(index_path: Path = DEFAULT_INDEX_PATH) -> "FaissVectorStore":
    """Return the process-wide store for an index file, loading it from disk only once.

    Ingestion and search share this instance so a search never reads a half-written file
    and two concurrent adds cannot hand out the same faiss_ids.
    """
    key = Path(index_path).resolve()
    with _stores_lock:
        store = _stores.get(key)
        if store is None:
            store = FaissVectorStore(index_path=key)
            _stores[key] = store
        return store


class FaissVectorStore:
    """FAISS HNSW index for 512-dim normalized vectors."""

    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
        dim: int = 512,
        m_hnsw: int = 32,
    ) -> None:
        self.index_path = Path(index_path)
        self.dim = dim
        self.m_hnsw = m_hnsw
        self._lock = threading.RLock()
        self.index: faiss.Index = self._load_or_create()
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = HNSW_EF_SEARCH

    def _load_or_create(self) -> faiss.Index:
        """Load existing FAISS index from disk, or initialize a new HNSW index."""
        if self.index_path.exists():
            try:
                idx = faiss.read_index(str(self.index_path))
                logger.info("Loaded existing FAISS index from %s with %d vectors", self.index_path, idx.ntotal)
                return idx
            except Exception as e:
                # Never let the next save() overwrite an unreadable index: set it aside so it can be recovered
                quarantined = self.index_path.with_name(f"{self.index_path.name}.corrupt-{int(time.time())}")
                try:
                    os.replace(self.index_path, quarantined)
                except OSError as move_err:
                    logger.error("Could not set aside unreadable FAISS index %s: %s", self.index_path, move_err)
                    raise RuntimeError(f"FAISS index {self.index_path} is unreadable and could not be moved aside") from e
                logger.error(
                    "Failed to read FAISS index from %s (%s); moved it to %s and starting an empty index",
                    self.index_path,
                    e,
                    quarantined,
                )

        logger.info("Initializing new FAISS IndexHNSWFlat(dim=%d, M=%d, METRIC_INNER_PRODUCT)", self.dim, self.m_hnsw)
        # Using METRIC_INNER_PRODUCT since embeddings are L2-normalized
        idx = faiss.IndexHNSWFlat(self.dim, self.m_hnsw, faiss.METRIC_INNER_PRODUCT)
        return idx

    @property
    def total_vectors(self) -> int:
        """Current number of vectors in index."""
        return self.index.ntotal

    def add_vectors(self, vectors: np.ndarray) -> List[int]:
        """Add batch of normalized float32 vectors to the index.
        
        Args:
            vectors: np.ndarray of shape (N, 512), dtype float32.
            
        Returns:
            List of integer faiss_ids corresponding to the added vectors.
        """
        if len(vectors) == 0:
            return []

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        # A NaN in the HNSW graph silently corrupts every later traversal; refuse it outright
        if not np.isfinite(vectors).all():
            raise ValueError("Refusing to index non-finite vectors")

        with self._lock:
            start_id = self.index.ntotal
            self.index.add(vectors)
            end_id = self.index.ntotal
            faiss_ids = list(range(start_id, end_id))
            self.save()
        logger.info("Added %d vectors to FAISS index (IDs %d..%d, total: %d)", len(faiss_ids), start_id, end_id - 1, self.total_vectors)
        return faiss_ids

    def save(self) -> None:
        """Atomically persist FAISS index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.index_path.parent / f"temp_{self.index_path.name}"
        
        try:
            faiss.write_index(self.index, str(temp_file))
            os.replace(temp_file, self.index_path)
            logger.debug("Persisted FAISS index to %s", self.index_path)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            logger.error("Failed to save FAISS index: %s", e)
            raise

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search nearest vectors by inner product (cosine similarity).
        
        Args:
            query_vector: np.ndarray of shape (1, 512) or (512,), dtype float32.
            top_k: Number of nearest neighbors to retrieve.
            
        Returns:
            Tuple of (scores, indices), where scores are cosine similarities and indices are faiss_ids.
        """
        if self.index.ntotal == 0:
            return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        with self._lock:
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(query_vector, k)
        return scores, indices
