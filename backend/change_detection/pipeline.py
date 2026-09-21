"""IRIS change detection — pipeline orchestrator.

run_change_detection(scene_a_id, scene_b_id) runs Phases 1-5 for one ordered pair (A older, B newer) and
records the outcome in the jobs / change_candidates tables. The job's `details` JSON is the decision trace:
which masking path ran, how the grids compared, the alignment method and its quality, the PIF tier, the K-means
centroids, the seasonal-filter outcome, and wall-clock time per phase.
"""

import gc
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import rasterio
from rasterio.enums import Resampling

import instrumentation

from catalog import changes as jobs
from catalog.database import get_scene, init_connection, init_schema
from change_detection import params
from change_detection.alignment import (
    AlignmentFailed,
    describe as describe_alignment,
    estimate_alignment,
    regrid_to_reference,
    same_tile_alignment,
    warp_mask,
)
from change_detection.detection import detect_changes
from change_detection.direction import classify_candidates
from change_detection.grouping import assign_groups, attach_geometry, merge_candidates
from change_detection.masking import build_validity
from change_detection.radiometry import fit_normalization
from change_detection.rasters import SceneReader, bands_summary, grid_signature, grids_match, mgrs_tile_id
from change_detection.scoring import finalize_candidates, score_candidates
from change_detection.seasonal import apply_seasonality

logger = logging.getLogger("iris.change.pipeline")

STAGING_DIR = Path("data/staging")


class GridMismatch(Exception):
    """The pair still does not share a pixel grid after warping B onto A's grid."""


class InsufficientEvidence(Exception):
    """Too little of the pair was observed in both scenes to say anything about change."""


def _scene_paths(scene: Dict[str, Any]) -> Dict[str, Optional[Path]]:
    """Analysis raster and SCL for a scene; a non-Sentinel-2 scene falls back to its display COG and no SCL."""
    analysis = Path(scene["analysis_cog_path"] or scene["cog_path"] or "")
    if not analysis.is_file():
        raise ValueError(f"Analysis raster for {scene['scene_id']} is missing on disk")
    scl = None
    if scene.get("scl_path"):
        scl = Path(scene["scl_path"])
        if not scl.is_file():
            raise ValueError(f"SCL raster for {scene['scene_id']} is missing on disk")
    return {"analysis": analysis, "scl": scl}


def _mask_kwargs(scene: Dict[str, Any]) -> Dict[str, Any]:
    """How to read a scene's mask: a Landsat mask is QA_PIXEL (in SCL codes); no mask but a cloud estimate is heuristic."""
    kind = "qa_pixel" if str(scene.get("sensor", "")).startswith("landsat") else "scl"
    return {"mask_kind": kind, "heuristic_cloud_pct": None if scene.get("scl_path") else scene.get("cloud_pct")}


_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/)[^\s'\"),;]+")


def _safe_message(exc: Exception) -> str:
    """Exception text for the analyst: type and reason, with any file-system path removed."""
    return f"{type(exc).__name__}: {_PATH_RE.sub('<path>', str(exc))}"[:300]


def _persist(job_id: int, details: Dict[str, Any]) -> None:
    conn = init_connection()
    try:
        jobs.update_job_details(conn, job_id, details)
    finally:
        conn.close()


def _finish(job_id: int, status: str, details: Dict[str, Any], error: Optional[str] = None, candidates=None) -> None:
    conn = init_connection()
    try:
        jobs.finish_job(conn, job_id, status, details, error=error, candidates=candidates)
    finally:
        conn.close()


def run_change_detection(scene_a_id: str, scene_b_id: str) -> Dict[str, Any]:
    """Run the full pipeline for one pair and return {"job_id", "status", "candidates", "error"}.

    A older than B is enforced (the pair is swapped if needed). Differencing is restricted to scenes of the same
    sensor, and a pair that already reached a final state is never processed again.
    """
    conn = init_connection()
    try:
        init_schema(conn)
        scene_a, scene_b = get_scene(conn, scene_a_id), get_scene(conn, scene_b_id)
        if scene_a is None or scene_b is None:
            raise ValueError("Both scenes must be in the catalog before change detection")
        if scene_a["sensor"] != scene_b["sensor"]:
            # Cross-sensor differencing would present sensor differences as change (no BRDF/bandpass harmonisation)
            raise ValueError("Change detection is restricted to scenes from the same sensor")
        date_a, date_b = scene_a["acquisition_date"], scene_b["acquisition_date"]
        if "unknown" in (date_a, date_b) or date_a == date_b:
            raise ValueError("Scenes need two different, known acquisition dates to be ordered")
        if date_a > date_b:
            scene_a, scene_b = scene_b, scene_a
        job = jobs.create_job(conn, scene_a["scene_id"], scene_b["scene_id"])
        if job["status"] in jobs.JOB_FINAL_STATES:
            logger.info("Pair %s -> %s already %s; not re-running", scene_a["scene_id"], scene_b["scene_id"], job["status"])
            return {"job_id": job["job_id"], "status": job["status"], "candidates": None, "error": job["error_message"]}
        jobs.mark_job_processing(conn, job["job_id"])
    finally:
        conn.close()

    return _execute(job["job_id"], scene_a, scene_b)


def _execute(job_id: int, scene_a: Dict[str, Any], scene_b: Dict[str, Any]) -> Dict[str, Any]:
    timer = instrumentation.PipelineTimer()
    timings: Dict[str, float] = {}  # seconds per phase, as the job's decision trace records them
    details: Dict[str, Any] = {
        "phase": "starting",
        "scene_a": {"scene_id": scene_a["scene_id"], "date": scene_a["acquisition_date"]},
        "scene_b": {"scene_id": scene_b["scene_id"], "date": scene_b["acquisition_date"]},
        "timings_s": timings,
    }
    work_dir = STAGING_DIR / f"change_{job_id}"
    det = None

    def sync_timings() -> None:
        timings.update({name: round(ms / 1000.0, 2) for name, ms in timer.to_dict().items() if name != "starting"})

    def phase(name: str) -> None:
        timer.begin(name)  # closes the previous phase
        sync_timings()
        details["phase"] = name
        _persist(job_id, details)
        logger.info("Job %d: %s", job_id, name)

    def finish(status: str, error: Optional[str] = None, candidates=None) -> Dict[str, Any]:
        timer.end()
        sync_timings()
        details["phase"] = "done"
        timings["total"] = round(timer.total_ms() / 1000.0, 2)
        _finish(job_id, status, details, error, candidates)
        instrumentation.record_change_detection(
            job_id, scene_a["scene_id"], scene_b["scene_id"], dict(timings), status, len(candidates or [])
        )
        return {"job_id": job_id, "status": status, "candidates": len(candidates or []), "error": error}

    try:
        paths_a, paths_b = _scene_paths(scene_a), _scene_paths(scene_b)
        work_dir.mkdir(parents=True, exist_ok=True)

        # ---- Phase 1: quality masking and CloudTrust ------------------------------------------------------
        phase("masking")
        val_a = build_validity(paths_a["analysis"], paths_a["scl"], **_mask_kwargs(scene_a))
        val_b = build_validity(paths_b["analysis"], paths_b["scl"], **_mask_kwargs(scene_b))
        details["masking"] = {
            "scene_a": {"source": val_a.source, "cloud_trust": round(val_a.cloud_trust, 4), "snow_fraction": round(val_a.snow_fraction, 4)},
            "scene_b": {"source": val_b.source, "cloud_trust": round(val_b.cloud_trust, 4), "snow_fraction": round(val_b.snow_fraction, 4)},
        }

        # ---- Phase 2: grid check FIRST, then mutual mask, then alignment ----------------------------------
        phase("grid_check")
        sig_a, sig_b = grid_signature(paths_a["analysis"]), grid_signature(paths_b["analysis"])
        with rasterio.open(str(paths_a["analysis"])) as sa, rasterio.open(str(paths_b["analysis"])) as sb:
            count_a, count_b, dtype_a, dtype_b = sa.count, sb.count, sa.dtypes[0], sb.dtypes[0]
        if (count_a, dtype_a) != (count_b, dtype_b):
            raise ValueError("The two scenes have different band layouts or data types and cannot be differenced")
        details["bands"] = bands_summary(count_a)

        analysis_b, valid_b = paths_b["analysis"], val_b.valid
        tile_a, tile_b = mgrs_tile_id(scene_a["scene_id"]), mgrs_tile_id(scene_b["scene_id"])
        same_tile = tile_a is not None and tile_a == tile_b
        if grids_match(sig_a, sig_b):
            details["grid"] = {"matched": True}
        else:
            # Same CRS is not enough; warp B onto A's grid, then verify the grids really are identical
            regridded = work_dir / "b_on_a_grid.tif"
            regrid_to_reference(paths_b["analysis"], paths_a["analysis"], regridded, Resampling.bilinear)
            regridded_scl = None
            if paths_b["scl"] is not None:
                regridded_scl = work_dir / "b_scl_on_a_grid.tif"
                regrid_to_reference(paths_b["scl"], paths_a["analysis"], regridded_scl, Resampling.nearest)  # categorical
            if not grids_match(sig_a, grid_signature(regridded)):
                raise GridMismatch("Scene B could not be brought onto scene A's pixel grid")
            analysis_b = regridded
            valid_b = build_validity(regridded, regridded_scl, **_mask_kwargs(scene_b)).valid
            details["grid"] = {"matched": False, "regridded": True, "scl_resampling": "nearest"}

        mutual = val_a.valid & valid_b
        coverage = float(mutual.mean())
        details["mutual_coverage"] = round(coverage, 4)
        if coverage < params.MIN_MUTUAL_COVERAGE:
            raise InsufficientEvidence(
                f"Mutual valid coverage {coverage:.1%} is below the {params.MIN_MUTUAL_COVERAGE:.0%} minimum"
            )

        phase("alignment")
        if same_tile and details["grid"]["matched"]:
            # Same MGRS tile = the same fixed pixel grid, and the grid check above just confirmed it. If that check
            # had failed, the tile ID would not be trusted and alignment would run as for any other pair.
            logger.info("Same MGRS tile, grid alignment verified by definition")
            alignment = same_tile_alignment(tile_a)
        else:
            with SceneReader(paths_a["analysis"]) as reader_a, SceneReader(analysis_b) as reader_b_raw:
                alignment = estimate_alignment(reader_a, reader_b_raw, mutual)
        details["alignment"] = describe_alignment(alignment)
        details["alignment"]["tiles"] = [tile_a, tile_b]

        # The warp moves B's pixels, so B's validity moves with it and the mutual mask is recomputed
        mutual = val_a.valid & warp_mask(valid_b, alignment.matrix)
        coverage = float(mutual.mean())
        details["mutual_coverage_after_alignment"] = round(coverage, 4)
        if coverage < params.MIN_MUTUAL_COVERAGE:
            raise InsufficientEvidence(
                f"Mutual valid coverage after alignment {coverage:.1%} is below the {params.MIN_MUTUAL_COVERAGE:.0%} minimum"
            )
        src_a, src_b = val_a.source, val_b.source
        cloud_trust_a, cloud_trust_b = val_a.cloud_trust, val_b.cloud_trust
        del val_a, val_b, valid_b
        gc.collect()

        with SceneReader(paths_a["analysis"]) as reader_a, SceneReader(analysis_b, alignment.matrix) as reader_b:
            # ---- Phase 3: radiometric normalisation -------------------------------------------------------
            phase("radiometry")
            norm = fit_normalization(reader_a, reader_b, mutual)
            details["radiometry"] = norm.describe()

            # ---- Phase 4: change detection ----------------------------------------------------------------
            phase("detection")
            det = detect_changes(reader_a, reader_b, mutual, norm, work_dir)
            details["detection"] = det.trace

        # ---- Grouping: which surviving blobs belong to one detection (mask dilation; geometry stays undilated) ----
        phase("grouping")
        details["grouping"] = assign_groups(det)

        # ---- Phase 5: scoring -----------------------------------------------------------------------------
        phase("scoring")
        with rasterio.open(str(paths_a["analysis"])) as src:
            transform, crs = src.transform, (src.crs.to_string() if src.crs else None)
        # A heuristic cloud estimate removed no pixel, so it reaches confidence here instead
        trust = min(v for v, src in ((cloud_trust_a, src_a), (cloud_trust_b, src_b)) if src == "heuristic") if "heuristic" in (src_a, src_b) else 1.0
        candidates, score_trace = score_candidates(
            det, mutual, alignment.quality, transform, crs, scene_a["scene_id"], scene_b["scene_id"], cloud_trust=trust
        )
        score_trace["cloud_trust_factor"] = round(trust, 4)
        details["scoring"] = score_trace

        # ---- Phase 4b: direction and refined change type for every surviving blob ---------------------------
        phase("direction")
        details["direction"] = classify_candidates(det, candidates)

        # ---- One detection per group: merge, apply the storage threshold, trace the outline ---------------
        phase("merging")
        candidates = finalize_candidates(merge_candidates(candidates, crs), score_trace)
        attach_geometry(det, candidates, transform, crs)
        details["grouping"]["detections"] = len(candidates)

        # ---- Seasonal persistence filter: is this vegetation drop just the time of year? -------------------
        phase("seasonality")
        details["seasonality"] = apply_seasonality(candidates, scene_b)
        return finish(jobs.JOB_COMPLETED, candidates=candidates)

    except InsufficientEvidence as e:
        logger.warning("Job %d: insufficient evidence: %s", job_id, e)
        return finish(jobs.JOB_INSUFFICIENT_EVIDENCE, str(e))
    except AlignmentFailed as e:
        logger.warning("Job %d: alignment failed: %s", job_id, e)
        return finish(jobs.JOB_ALIGNMENT_FAILED, str(e))
    except GridMismatch as e:
        logger.warning("Job %d: grid mismatch: %s", job_id, e)
        return finish(jobs.JOB_GRID_MISMATCH, str(e))
    except Exception as e:
        logger.exception("Job %d failed", job_id)
        return finish(jobs.JOB_FAILED, _safe_message(e))
    finally:
        # Release the memory-mapped intermediates before deleting them (Windows will not delete an open file)
        det = None
        gc.collect()
        shutil.rmtree(work_dir, ignore_errors=True)
