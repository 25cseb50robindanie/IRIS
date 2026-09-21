"""IRIS Pipeline Status — step-by-step progress of an import, from format detection to change detection.

Ingest is synchronous up to COG conversion and then continues in a background job (embedding, then change
detection), so no single request can report progress. The client picks a `pipeline_id`, sends it with the ingest
request, and polls GET /api/pipeline/{id} from the start; the same record carries on through the background job.
Progress is held in memory: it is transient by nature, and durable outcomes live in the catalog (scenes, tiles,
jobs, change_candidates).
"""

import re
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

PENDING, ACTIVE, DONE, FAILED, SKIPPED = "pending", "active", "done", "failed", "skipped"
RUNNING, FINISHED, ERROR = "running", "done", "failed"

PIPELINE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
MAX_TRACKED = 25  # oldest pipelines are dropped beyond this

# key -> label for every step an import can have
LABELS: Dict[str, str] = {
    "detect": "Detecting format",
    "bands": "Extracting bands (B02, B03, B04, B08)",
    "scl": "Reading SCL band",
    "rgb": "Building RGB composite",
    "cog": "Converting to COG",
    "crops": "Generating embedding crops",
    "embed": "Running RemoteCLIP",
    "index": "Indexing",
    "overlap": "Checking for overlapping scenes",
    "change": "Running change detection",
}
SAFE_STEPS = ["bands", "scl", "rgb", "cog", "crops", "embed", "index", "overlap", "change"]
SINGLE_FILE_STEPS = ["cog", "crops", "embed", "index", "overlap", "change"]


def new_pipeline_id() -> str:
    return uuid.uuid4().hex


class _Step:
    def __init__(self, key: str) -> None:
        self.key = key
        self.label = LABELS[key]
        self.state = PENDING
        self.detail: Optional[str] = None
        self.done: Optional[int] = None
        self.total: Optional[int] = None
        self.started: Optional[float] = None  # wall-clock start of the step (first time it was active)
        self.ended: Optional[float] = None

    def touch(self) -> None:
        """Keep the step's start/end times in step with its state."""
        now = time.monotonic()
        if self.state == ACTIVE and self.started is None:
            self.started = now
        if self.state in (DONE, FAILED, SKIPPED) and self.started is not None and self.ended is None:
            self.ended = now


class _Pipeline:
    def __init__(self, pipeline_id: str, name: str) -> None:
        self.pipeline_id = pipeline_id
        self.name = name
        self.scene_id: Optional[str] = None
        self.state = RUNNING
        self.message: Optional[str] = None
        self.steps: List[_Step] = [_Step("detect")]
        self.started = time.monotonic()
        self.ended: Optional[float] = None

    def step(self, key: str) -> _Step:
        for s in self.steps:
            if s.key == key:
                return s
        s = _Step(key)  # a step that was not planned (e.g. crops on a path that skipped planning) is appended
        self.steps.append(s)
        return s


class PipelineTracker:
    """Thread-safe registry of import pipelines."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pipelines: "OrderedDict[str, _Pipeline]" = OrderedDict()
        self._by_scene: Dict[str, str] = {}

    def start(self, pipeline_id: str, name: str = "") -> None:
        with self._lock:
            pipe = _Pipeline(pipeline_id, name)
            pipe.step("detect").state = ACTIVE
            pipe.step("detect").touch()
            self._pipelines[pipeline_id] = pipe
            while len(self._pipelines) > MAX_TRACKED:
                _, old = self._pipelines.popitem(last=False)
                if old.scene_id and self._by_scene.get(old.scene_id) == old.pipeline_id:
                    del self._by_scene[old.scene_id]

    def plan(self, pipeline_id: str, keys: List[str]) -> None:
        """Declare the steps still to come so the analyst sees the whole pipeline, not just the current stage."""
        with self._lock:
            pipe = self._pipelines.get(pipeline_id)
            if pipe:
                for key in keys:
                    pipe.step(key)

    def set_scene(self, pipeline_id: str, scene_id: str) -> None:
        with self._lock:
            pipe = self._pipelines.get(pipeline_id)
            if pipe:
                pipe.scene_id = scene_id
                self._by_scene[scene_id] = pipeline_id

    def _update(self, pipeline_id: Optional[str], key: str, **fields: Any) -> None:
        if not pipeline_id:
            return
        with self._lock:
            pipe = self._pipelines.get(pipeline_id)
            if pipe is None:
                return
            step = pipe.step(key)
            for name, value in fields.items():
                setattr(step, name, value)
            step.touch()

    def begin(self, pid: Optional[str], key: str, detail: Optional[str] = None) -> None:
        self._update(pid, key, state=ACTIVE, detail=detail)

    def progress(self, pid: Optional[str], key: str, done: int, total: int, detail: Optional[str] = None) -> None:
        self._update(pid, key, state=ACTIVE, done=done, total=total, detail=detail)

    def complete(self, pid: Optional[str], key: str, detail: Optional[str] = None) -> None:
        self._update(pid, key, state=DONE, detail=detail)

    def skip(self, pid: Optional[str], key: str, detail: Optional[str] = None) -> None:
        self._update(pid, key, state=SKIPPED, detail=detail)

    def fail_step(self, pid: Optional[str], key: str, error: str) -> None:
        self._update(pid, key, state=FAILED, detail=error)

    def fail_active(self, pid: Optional[str], error: str, end: bool = True) -> bool:
        """Mark whichever step was running as failed (returns whether one was); end=True also ends the import."""
        if not pid:
            return False
        with self._lock:
            pipe = self._pipelines.get(pid)
            if pipe is None:
                return False
            active = next((s for s in pipe.steps if s.state == ACTIVE), None)
            if active is not None:
                active.state, active.detail = FAILED, error
                active.touch()
            if end:
                pipe.state, pipe.message, pipe.ended = ERROR, f"Import failed — {error}", time.monotonic()
            return active is not None

    def finish(self, pid: Optional[str], message: str) -> None:
        if not pid:
            return
        with self._lock:
            pipe = self._pipelines.get(pid)
            if pipe is None or pipe.state == ERROR:
                return
            for s in pipe.steps:  # anything still waiting was never reached
                if s.state in (PENDING, ACTIVE):
                    s.state = SKIPPED
            pipe.state, pipe.message, pipe.ended = FINISHED, message, time.monotonic()

    def timings(self, pid: Optional[str]) -> Dict[str, float]:
        """Seconds spent in every step that ran, keyed by step key (skipped and never-started steps are absent)."""
        with self._lock:
            pipe = self._pipelines.get(pid or "")
            if pipe is None:
                return {}
            return {
                s.key: round(s.ended - s.started, 3) for s in pipe.steps if s.started is not None and s.ended is not None
            }

    def elapsed(self, pid: Optional[str]) -> float:
        with self._lock:
            pipe = self._pipelines.get(pid or "")
            if pipe is None:
                return 0.0
            return (pipe.ended if pipe.ended is not None else time.monotonic()) - pipe.started

    def snapshot(self, ident: str) -> Optional[Dict[str, Any]]:
        """State by pipeline id, or by scene id (the latest pipeline that imported that scene)."""
        with self._lock:
            pipe = self._pipelines.get(ident) or self._pipelines.get(self._by_scene.get(ident, ""))
            if pipe is None:
                return None
            end = pipe.ended if pipe.ended is not None else time.monotonic()
            return {
                "pipeline_id": pipe.pipeline_id,
                "scene_id": pipe.scene_id,
                "name": pipe.name,
                "state": pipe.state,
                "message": pipe.message,
                "elapsed_s": round(end - pipe.started, 1),
                "steps": [
                    {
                        "key": s.key,
                        "label": s.label,
                        "state": s.state,
                        "detail": s.detail,
                        "done": s.done,
                        "total": s.total,
                    }
                    for s in pipe.steps
                ],
            }


pipeline_tracker = PipelineTracker()


class StepModel(BaseModel):
    key: str
    label: str
    state: str  # pending | active | done | failed | skipped
    detail: Optional[str] = None
    done: Optional[int] = None
    total: Optional[int] = None


class PipelineModel(BaseModel):
    pipeline_id: str
    scene_id: Optional[str] = None
    name: str = ""
    state: str  # running | done | failed
    message: Optional[str] = None
    elapsed_s: float = 0.0
    steps: List[StepModel]


router = APIRouter(prefix="/api", tags=["pipeline"])


@router.get("/pipeline/{ident}", response_model=PipelineModel)
def get_pipeline(ident: str) -> PipelineModel:
    """Progress of an import, addressed by its pipeline id or by the scene id it produced."""
    if len(ident) > 200:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid pipeline id")
    snap = pipeline_tracker.snapshot(ident)
    if snap is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such pipeline")
    return PipelineModel(**snap)
