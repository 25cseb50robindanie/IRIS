"""IRIS Embedding Progress — thread-safe, in-memory status of background embedding jobs."""

import threading
import time
from typing import Any, Dict, Optional

QUEUED = "queued"
TILING = "tiling"
EMBEDDING = "embedding"
INDEXING = "indexing"
READY = "ready"
FAILED = "failed"

TERMINAL_STATES = frozenset({READY, FAILED})


class _Job:
    """Mutable progress record for one scene's embedding job."""

    def __init__(self, scene_id: str) -> None:
        self.scene_id = scene_id
        self.state = QUEUED
        self.crops_generated = 0
        self.crops_total = 0
        self.crops_embedded = 0
        self.tiles_count = 0
        self.device: Optional[str] = None
        self.error: Optional[str] = None
        self.started_at = time.monotonic()
        self.finished_at: Optional[float] = None
        self.embed_started_at: Optional[float] = None
        self.embed_finished_at: Optional[float] = None


class ProgressTracker:
    """Registry of embedding job progress, safe to update from worker threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, _Job] = {}

    def start(self, scene_id: str) -> None:
        """Register a fresh job in the queued state, replacing any earlier record."""
        with self._lock:
            self._jobs[scene_id] = _Job(scene_id)

    def update(self, scene_id: str, **fields: Any) -> None:
        """Set fields on a job; a state change also stamps the timing used for throughput."""
        with self._lock:
            job = self._jobs.get(scene_id)
            if job is None:
                return
            state = fields.get("state")
            now = time.monotonic()
            if state == EMBEDDING and job.embed_started_at is None:
                job.embed_started_at = now
            if state in (INDEXING, READY, FAILED) and job.embed_started_at and job.embed_finished_at is None:
                job.embed_finished_at = now
            if state in TERMINAL_STATES:
                job.finished_at = now
            for key, value in fields.items():
                setattr(job, key, value)

    def forget(self, scene_id: str) -> None:
        with self._lock:
            self._jobs.pop(scene_id, None)

    def fail(self, scene_id: str, error: str) -> None:
        self.update(scene_id, state=FAILED, error=error)

    def snapshot(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """Return a plain-dict copy of the job's progress, or None if it isn't tracked."""
        with self._lock:
            job = self._jobs.get(scene_id)
            if job is None:
                return None
            end = job.finished_at if job.finished_at is not None else time.monotonic()
            crops_per_sec: Optional[float] = None
            if job.embed_started_at is not None and job.crops_embedded > 0:
                embed_end = job.embed_finished_at if job.embed_finished_at is not None else time.monotonic()
                span = embed_end - job.embed_started_at
                if span > 0:
                    crops_per_sec = round(job.crops_embedded / span, 2)
            return {
                "scene_id": job.scene_id,
                "state": job.state,
                "crops_generated": job.crops_generated,
                "crops_total": job.crops_total,
                "crops_embedded": job.crops_embedded,
                "tiles_count": job.tiles_count,
                "device": job.device,
                "error": job.error,
                "elapsed_s": round(end - job.started_at, 1),
                "crops_per_sec": crops_per_sec,
            }


embedding_progress = ProgressTracker()
