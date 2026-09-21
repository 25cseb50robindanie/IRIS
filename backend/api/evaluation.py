"""IRIS evaluation API — the manifest of build times, storage, index size, query latency and hardware."""

from typing import Any, Dict

from fastapi import APIRouter

import instrumentation

router = APIRouter(prefix="/api/eval", tags=["evaluation"])


@router.get("/manifest")
def eval_manifest() -> Dict[str, Any]:
    """The evaluation manifest (also kept up to date at data/eval_manifest.json), with storage and index measured now."""
    return instrumentation.manifest()
