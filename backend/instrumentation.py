"""IRIS evaluation instrumentation — the numbers an evaluation report needs, recorded as the work happens.

Build time and query latency cannot be reconstructed afterwards (architecture.md, "Evaluation Instrumentation"), so
every ingestion, every change-detection job and every query reports here, and `data/eval_manifest.json` is rewritten
after each one. The manifest carries:

  * per-stage wall-clock time of each ingestion and each change-detection job
  * storage footprint (COGs, crops, FAISS index, SQLite catalog)
  * index statistics (scenes, tiles, FAISS vectors, summed scene footprints)
  * query latency (count, mean, p50/p95/p99, max), separately for text search and find-similar
  * hardware and the versions of the libraries that produced the numbers

Recording never raises into the caller: a failed measurement must not fail an import or a search.
"""

import json
import logging
import os
import platform
import sys
import threading
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("iris.instrumentation")

MANIFEST_PATH = Path("data/eval_manifest.json")
MAX_OPERATION_RECORDS = 200  # oldest ingestions / jobs are dropped beyond this
MAX_LATENCY_SAMPLES = 1000  # per query kind

# The pipeline's own phase names, named as the evaluation report calls them
CHANGE_STAGE_NAMES = {"radiometry": "normalisation", "direction": "classification"}
INGEST_STAGE_NAMES = {
    "detect": "format_detection",
    "bands": "band_extraction",
    "scl": "scl_masking",
    "rgb": "rgb_composite",
    "cog": "cog_conversion",
    "crops": "crop_generation",
    "embed": "embedding",
    "index": "faiss_insertion",
    "overlap": "overlap_check",
    "change": "change_detection",
}

QUERY_TEXT = "text_search"
QUERY_SIMILAR = "find_similar"

_lock = threading.RLock()
_loaded_for: Optional[Path] = None
_state: Dict[str, Any] = {}
_hardware: Optional[Dict[str, Any]] = None
_cached_sections: Dict[str, Any] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _empty_state() -> Dict[str, Any]:
    return {"ingestions": [], "change_detection": [], "query_samples_ms": {}}


def _ensure_loaded() -> None:
    """Pick up what an earlier run of the server already recorded (and follow the working directory: tests move it)."""
    global _loaded_for, _state, _cached_sections
    path = MANIFEST_PATH.resolve()
    if _loaded_for == path:
        return
    _state, _cached_sections = _empty_state(), {}
    if path.is_file():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            _state["ingestions"] = list(old.get("ingestions", []))[-MAX_OPERATION_RECORDS:]
            _state["change_detection"] = list(old.get("change_detection", []))[-MAX_OPERATION_RECORDS:]
            _state["query_samples_ms"] = {
                kind: list(v.get("samples_ms", []))[-MAX_LATENCY_SAMPLES:] for kind, v in old.get("queries", {}).items()
            }
        except Exception:
            logger.warning("Could not read the existing evaluation manifest; starting a new one")
    _loaded_for = path


# ---- hardware and software (captured once per run) ----------------------------------------------------------------


def _cpu_model() -> str:
    try:
        if sys.platform == "win32":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            for line in cpuinfo.read_text(errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _gpu_info() -> Dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {"available": True, "name": props.name, "memory_gb": round(props.total_memory / 1024**3, 1)}
        return {"available": False, "name": None, "note": "CUDA not available; inference runs on the CPU"}
    except Exception as e:
        return {"available": False, "name": None, "note": f"torch not usable: {type(e).__name__}"}


def _version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except Exception:
        return None


def hardware() -> Dict[str, Any]:
    global _hardware
    with _lock:
        if _hardware is None:
            ram = None
            cores = os.cpu_count()
            try:
                import psutil

                ram = round(psutil.virtual_memory().total / 1024**3, 1)
                cores = psutil.cpu_count(logical=False) or cores
            except Exception:
                pass
            _hardware = {
                "cpu_model": _cpu_model(),
                "cpu_cores_physical": cores,
                "cpu_threads": os.cpu_count(),
                "ram_total_gb": ram,
                "gpu": _gpu_info(),
                "os": platform.platform(),
                "python": platform.python_version(),
            }
        return _hardware


def software() -> Dict[str, Optional[str]]:
    return {
        pkg: _version(pkg)
        for pkg in ("fastapi", "rasterio", "numpy", "scipy", "opencv-python-headless", "faiss-cpu", "torch", "open_clip_torch", "mgrs")
    }


# ---- storage and index -------------------------------------------------------------------------------------------


def _tree_bytes(path: Path) -> int:
    from ingestion.loader import io_path

    total = 0
    stack = [io_path(path)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat().st_size
        except OSError:
            continue
    return total


def _file_bytes(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def storage_footprint() -> Dict[str, Any]:
    from catalog.database import DB_PATH
    from embedding.index import DEFAULT_INDEX_PATH

    sqlite = sum(_file_bytes(Path(str(DB_PATH) + suffix)) for suffix in ("", "-wal", "-shm"))
    sizes = {
        "cogs_bytes": _tree_bytes(Path("data/cogs")),
        "crops_bytes": _tree_bytes(Path("data/crops")),
        "faiss_index_bytes": _file_bytes(DEFAULT_INDEX_PATH),
        "sqlite_bytes": sqlite,
    }
    sizes["total_bytes"] = sum(sizes.values())
    sizes["total_mb"] = round(sizes["total_bytes"] / 1024**2, 2)
    return sizes


def index_stats() -> Dict[str, Any]:
    from catalog.database import count_tiles, init_connection, init_schema
    from embedding.index import get_vector_store

    conn = init_connection()
    try:
        init_schema(conn)
        scenes = int(conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0])
        tiles = count_tiles(conn)
        footprints = conn.execute("SELECT min_lon, max_lon, min_lat, max_lat FROM scene_footprints").fetchall()
        candidates = int(conn.execute("SELECT COUNT(*) FROM change_candidates").fetchone()[0])
    finally:
        conn.close()
    area = 0.0
    for min_lon, max_lon, min_lat, max_lat in footprints:
        mid = np.radians((min_lat + max_lat) / 2.0)
        area += (max_lon - min_lon) * 111.320 * float(np.cos(mid)) * (max_lat - min_lat) * 110.574
    return {
        "scenes": scenes,
        "tiles": tiles,
        "faiss_vectors": int(get_vector_store().total_vectors),
        "change_candidates": candidates,
        # Bounding boxes, summed, so overlapping scenes are counted twice: an upper bound, not a deduplicated area
        "scene_footprints_km2_sum": round(area, 1),
    }


# ---- recording ---------------------------------------------------------------------------------------------------


def _percentiles(samples: List[float]) -> Dict[str, Any]:
    if not samples:
        return {"count": 0}
    a = np.asarray(samples, dtype=np.float64)
    return {
        "count": int(a.size),
        "mean_ms": round(float(a.mean()), 1),
        "p50_ms": round(float(np.percentile(a, 50)), 1),
        "p95_ms": round(float(np.percentile(a, 95)), 1),
        "p99_ms": round(float(np.percentile(a, 99)), 1),
        "max_ms": round(float(a.max()), 1),
    }


def _document(refresh_sections: bool, last_operation: str) -> Dict[str, Any]:
    if refresh_sections or not _cached_sections:
        _cached_sections["storage"] = storage_footprint()
        _cached_sections["index"] = index_stats()
    return {
        "schema": "iris.eval_manifest/1",
        "generated_at": _now(),
        "last_operation": last_operation,
        "hardware": hardware(),
        "software": software(),
        "storage": _cached_sections["storage"],
        "index": _cached_sections["index"],
        "ingestions": _state["ingestions"],
        "change_detection": _state["change_detection"],
        "queries": {
            kind: {**_percentiles(samples), "samples_ms": samples}
            for kind, samples in _state["query_samples_ms"].items()
        },
    }


def _write(doc: Dict[str, Any]) -> None:
    path = MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # readers never see a half-written manifest


def _commit(operation: str, refresh_sections: bool = True) -> None:
    try:
        _write(_document(refresh_sections, operation))
    except Exception:
        logger.exception("Could not update the evaluation manifest after %s", operation)


def record_ingestion(
    scene_id: Optional[str], name: str, step_seconds: Dict[str, float], total_s: float, outcome: str, tiles: int = 0
) -> None:
    """One import: seconds per stage (keyed by pipeline step), the total, and how it ended."""
    try:
        with _lock:
            _ensure_loaded()
            _state["ingestions"].append(
                {
                    "recorded_at": _now(),
                    "scene_id": scene_id,
                    "source": name,
                    "outcome": outcome,
                    "tiles_indexed": tiles,
                    "total_s": round(total_s, 2),
                    "stages_s": {INGEST_STAGE_NAMES.get(k, k): round(v, 2) for k, v in step_seconds.items()},
                }
            )
            del _state["ingestions"][:-MAX_OPERATION_RECORDS]
            _commit(f"ingestion:{scene_id}")
    except Exception:
        logger.exception("Could not record ingestion timings")


def record_change_detection(
    job_id: int, scene_a_id: str, scene_b_id: str, timings_s: Dict[str, float], status: str, candidates: int
) -> None:
    """One change-detection job: seconds per phase, from the job's own trace."""
    try:
        with _lock:
            _ensure_loaded()
            stages = {CHANGE_STAGE_NAMES.get(k, k): v for k, v in timings_s.items() if k != "total"}
            _state["change_detection"].append(
                {
                    "recorded_at": _now(),
                    "job_id": job_id,
                    "scene_a_id": scene_a_id,
                    "scene_b_id": scene_b_id,
                    "status": status,
                    "detections": candidates,
                    "total_s": timings_s.get("total"),
                    "stages_s": stages,
                }
            )
            del _state["change_detection"][:-MAX_OPERATION_RECORDS]
            _commit(f"change_detection:{job_id}")
    except Exception:
        logger.exception("Could not record change-detection timings")


def record_query(kind: str, seconds: float) -> None:
    """Latency of one query, from the request reaching the handler to the response being ready."""
    try:
        with _lock:
            _ensure_loaded()
            samples = _state["query_samples_ms"].setdefault(kind, [])
            samples.append(round(seconds * 1000.0, 1))
            del samples[:-MAX_LATENCY_SAMPLES]
            _commit(f"query:{kind}", refresh_sections=False)  # storage does not change with a query
    except Exception:
        logger.exception("Could not record query latency")


def note_change(operation: str) -> None:
    """Something changed the catalog or the files (a scene was deleted): refresh storage and index figures."""
    try:
        with _lock:
            _ensure_loaded()
            _commit(operation)
    except Exception:
        logger.exception("Could not refresh the evaluation manifest")


def manifest() -> Dict[str, Any]:
    """The current manifest, with storage and index figures measured now (and the file brought up to date)."""
    with _lock:
        _ensure_loaded()
        doc = _document(True, "manifest_requested")
        try:
            _write(doc)
        except Exception:
            logger.exception("Could not write the evaluation manifest")
        return doc


class Stopwatch:
    """`with Stopwatch() as t: ...` then `t.seconds`."""

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._start
