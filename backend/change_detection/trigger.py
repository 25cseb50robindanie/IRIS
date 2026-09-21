"""IRIS change detection — pairing and triggering.

After a scene is ingested, find overlapping scenes from the SAME sensor via the footprint R-Tree and run change
detection for the ordered pairs that result. Pairing follows the architecture's default: the nearest chronological
prior observation (and, if the new scene is older than something already catalogued, the nearest later one) rather
than every overlapping scene, which would grow combinatorially.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from catalog import changes as jobs
from catalog.database import find_overlapping_scenes, get_scene, init_connection, init_schema
from change_detection.ablation import run_ablation
from change_detection.pipeline import run_change_detection

logger = logging.getLogger("iris.change.trigger")

# A pair whose footprints barely touch cannot reach the mutual-coverage floor; don't even enqueue it
MIN_FOOTPRINT_OVERLAP = 0.30


def _known(date: str) -> bool:
    return bool(date) and date != "unknown"


def plan_pairs(scene_id: str) -> List[Tuple[str, str]]:
    """Ordered (A older, B newer) pairs this scene should be differenced in, and the jobs created for them."""
    conn = init_connection()
    try:
        init_schema(conn)
        scene = get_scene(conn, scene_id)
        if scene is None or not _known(scene["acquisition_date"]):
            logger.info("Scene %s has no usable acquisition date; skipping change detection", scene_id)
            return []

        others = [
            o
            for o in find_overlapping_scenes(conn, scene_id)
            if _known(o["acquisition_date"]) and o["overlap_fraction"] >= MIN_FOOTPRINT_OVERLAP
        ]
        this_date = scene["acquisition_date"]
        priors = [o for o in others if o["acquisition_date"] < this_date]
        later = [o for o in others if o["acquisition_date"] > this_date]
        # Same date means an adjacent tile of one pass, not a repeat observation, so it is never paired

        pairs: List[Tuple[str, str]] = []
        if priors:
            pairs.append((max(priors, key=lambda o: o["acquisition_date"])["scene_id"], scene_id))
        if later:
            pairs.append((scene_id, min(later, key=lambda o: o["acquisition_date"])["scene_id"]))

        for a_id, b_id in pairs:
            jobs.create_job(conn, a_id, b_id)  # idempotent: UNIQUE(job_type, scene_a_id, scene_b_id)
        return pairs
    finally:
        conn.close()


class Reporter:
    """Receives progress events while a scene is being paired and analysed. Every method is optional."""

    def overlap_check_started(self) -> None:
        pass

    def pairs_found(self, scene_id: str, pairs: List[Tuple[str, str]]) -> None:
        pass

    def pair_started(self, scene_id: str, a_id: str, b_id: str) -> None:
        pass

    def pair_finished(self, scene_id: str, a_id: str, b_id: str, result: Dict[str, Any]) -> None:
        pass


def schedule_change_detection(scene_id: str, reporter: Optional[Reporter] = None) -> List[Dict[str, Any]]:
    """Create jobs for this scene's overlapping same-sensor pairs and run them synchronously, in order."""
    rep = reporter or Reporter()
    rep.overlap_check_started()
    pairs = plan_pairs(scene_id)
    rep.pairs_found(scene_id, pairs)

    results: List[Dict[str, Any]] = []
    for a_id, b_id in pairs:
        rep.pair_started(scene_id, a_id, b_id)
        try:
            result = run_change_detection(a_id, b_id)
        except Exception as e:
            logger.exception("Change detection for %s -> %s could not start", a_id, b_id)
            result = {"job_id": None, "status": "failed", "candidates": 0, "error": f"{type(e).__name__}"}
        result = {**result, "scene_a_id": a_id, "scene_b_id": b_id}
        results.append(result)
        rep.pair_finished(scene_id, a_id, b_id, result)
    return results


def run_ablations(results: List[Dict[str, Any]]) -> int:
    """Run the ablation for every pair that has just completed. Returns how many succeeded.

    Kept apart from schedule_change_detection so the import can report "complete" first: the ablation is for the analyst
    who wants to see what the suppression stack removes, not something the import waits on. A failure is recorded on the
    job and logged; it never affects the analysis it belongs to.
    """
    done = 0
    for r in results:
        if r.get("status") != "completed" or r.get("candidates") is None:  # None: an earlier run, not re-run now
            continue
        try:
            run_ablation(r["scene_a_id"], r["scene_b_id"])
            done += 1
        except Exception:
            logger.warning("Ablation for %s -> %s did not complete", r["scene_a_id"], r["scene_b_id"])
    return done


def reconcile_interrupted_jobs() -> int:
    """Startup sweep: a job still 'processing' was interrupted by a shutdown, so put it back in the queue."""
    conn = init_connection()
    try:
        init_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute("UPDATE jobs SET status='queued' WHERE job_type='change_detection' AND status='processing'")
        conn.commit()
        if cur.rowcount:
            logger.warning("Reset %d interrupted change-detection job(s) to queued", cur.rowcount)
        return cur.rowcount
    finally:
        conn.close()
