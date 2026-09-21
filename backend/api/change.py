"""IRIS Change API Router — change candidates, pair jobs and analyst review decisions."""

import getpass
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from catalog import changes as store
from catalog.database import get_scene, init_connection, init_schema
from change_detection import params
from api.provenance import analysed_area, display_bounds, intersect, trace_lines
from change_detection.direction import CHANGE_TYPES, DIRECTIONS
from ingestion.loader import read_raster_bounds

logger = logging.getLogger("iris.api.change")
router = APIRouter(prefix="/api", tags=["change"])


class CandidateSummary(BaseModel):
    """One change candidate, as listed."""
    candidate_id: int
    job_id: int
    scene_a_id: str
    scene_b_id: str
    scene_a_date: Optional[str] = None
    scene_b_date: Optional[str] = None
    bounds: List[float]  # [min_lon, min_lat, max_lon, max_lat] in EPSG:4326 for map fitting
    change_type: Optional[str] = None
    direction: Optional[str] = None
    confidence: float
    area_px: Optional[int] = None
    area_ha: Optional[float] = None  # hectares, from the raster's pixel size
    mean_dndvi: Optional[float] = None
    review_status: str  # pending | confirmed | rejected
    mgrs: Optional[str] = None  # 10-digit MGRS reference of the detection's centroid
    centroid: Optional[List[float]] = None  # [lon, lat] of the changed pixels (None before it was traced)
    sub_blobs: int = 1  # change blobs merged into this detection
    seasonality_status: Optional[str] = None  # seasonal | anomalous | unverified; None when the persistence check did not apply


class AnalysedArea(BaseModel):
    """The ground the analysed pairs cover, for negative evidence."""
    bounds: Optional[List[float]] = None  # [min_lon, min_lat, max_lon, max_lat]
    mgrs: Optional[str] = None


class PairInfo(BaseModel):
    """One analysed pair of scenes (a change-detection job) and how many candidates it produced."""
    job_id: int
    scene_a_id: str
    scene_b_id: str
    scene_a_date: Optional[str] = None
    scene_b_date: Optional[str] = None
    status: str
    candidates: int  # stored candidates (confidence >= the storage threshold)
    found: int  # blobs that passed the minimum mapping unit, before the confidence threshold
    valid_coverage: Optional[float] = None  # share of the pair's footprint observed in both scenes


class CandidateList(BaseModel):
    total: int  # stored candidates in scope (the selected pair, or all pairs), before the filters below
    found: int  # candidates found before the storage threshold dropped the weak ones
    matching: int  # stored candidates that pass the confidence and type filters
    shown: int  # what is returned: matching, capped per pair at display_cap
    display_cap: int
    excluded_by_confidence: int  # weak candidates left out (by the slider, or dropped at storage)
    filtered_by_confidence: bool
    candidates: List[CandidateSummary]
    pairs: List[PairInfo]
    review_counts: Dict[str, int] = {"pending": 0, "confirmed": 0, "rejected": 0}  # stored candidates in scope, by decision
    thresholds: Dict[str, Any] = {}  # what 'no significant change' was measured against
    area: AnalysedArea = AnalysedArea()  # where the analysed pairs overlap


class SceneRef(BaseModel):
    """A scene as needed to render it in the before/after view."""
    scene_id: str
    acquisition_date: str
    sensor: str
    cog_url: str
    bounds_wgs84: Optional[List[float]] = None
    cloud_pct: Optional[float] = None


class ConfidenceTerm(BaseModel):
    value: float  # the term, in [0, 1]
    weight: float
    contribution: float  # value * weight


class ReviewEntry(BaseModel):
    review_id: int
    decision: str
    analyst_id: Optional[str] = None
    notes: Optional[str] = None
    reviewed_at: str


class TraceLine(BaseModel):
    key: str
    label: str
    text: str


class CandidateDetail(CandidateSummary):
    """Full detail for one candidate: both scenes, the confidence breakdown, and the decision trace."""
    bounds_native: List[float]
    display_bounds: List[float]  # 4x the box, clamped to the scene extent: the window the before/after view opens on
    processing_details: List[TraceLine] = []  # the decision trace as sentences, in pipeline order
    scene_a: SceneRef
    scene_b: SceneRef
    confidence_breakdown: Dict[str, ConfidenceTerm]  # the four weighted terms; their sum x confidence_factor = confidence
    confidence_factor: float = 1.0  # what the seasonal persistence filter multiplied the sum by (1.0: it did not apply)
    direction_evidence: Optional[Dict[str, Any]] = None  # what each date looked like, and the rule that named the change
    seasonality: Optional[Dict[str, Any]] = None  # the persistence filter's evidence (prior years, mean and spread of NDVI)
    terrain_is_placeholder: bool = True
    decision_trace: Dict[str, Any] = {}
    reviews: List[ReviewEntry] = []


class ReviewRequest(BaseModel):
    decision: Literal["confirmed", "rejected"]
    notes: Optional[str] = Field(default=None, max_length=1000)


class JobSummary(BaseModel):
    job_id: int
    scene_a_id: str
    scene_b_id: str
    scene_a_date: Optional[str] = None
    scene_b_date: Optional[str] = None
    status: str  # queued | processing | completed | failed | insufficient_evidence | alignment_failed | grid_mismatch
    phase: Optional[str] = None
    error: Optional[str] = None
    candidates: int = 0


class JobList(BaseModel):
    jobs: List[JobSummary]


def _summary(c: Dict[str, Any]) -> CandidateSummary:
    return CandidateSummary(
        candidate_id=c["candidate_id"],
        job_id=c["job_id"],
        scene_a_id=c["scene_a_id"],
        scene_b_id=c["scene_b_id"],
        scene_a_date=c["scene_a_date"],
        scene_b_date=c["scene_b_date"],
        bounds=[c["min_lon"], c["min_lat"], c["max_lon"], c["max_lat"]],
        change_type=c["change_type"],
        direction=c["direction"],
        confidence=c["confidence"],
        area_px=c["area_px"],
        area_ha=c["area_ha"],
        mean_dndvi=c["mean_dndvi"],
        review_status=c["review_status"],
        mgrs=c.get("mgrs_ref"),
        centroid=[c["centroid_lon"], c["centroid_lat"]] if c.get("centroid_lon") is not None else None,
        sub_blobs=c.get("sub_blobs") or 1,
        seasonality_status=c.get("seasonality_status"),
    )


def _scene_ref(conn, scene_id: str) -> SceneRef:
    scene = get_scene(conn, scene_id)
    if scene is None or not scene["cog_path"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Scene {scene_id} is no longer in the catalog")
    cog = Path(scene["cog_path"])
    bounds_wgs84 = None
    if cog.is_file():
        try:
            bounds_wgs84 = read_raster_bounds(cog)[1]
        except Exception:
            logger.warning("Could not read bounds of %s", cog.name)
    return SceneRef(
        scene_id=scene["scene_id"],
        acquisition_date=scene["acquisition_date"],
        sensor=scene["sensor"],
        cog_url=f"file:///{cog.resolve().as_posix()}",
        bounds_wgs84=bounds_wgs84,
        cloud_pct=scene["cloud_pct"],
    )


DETECTION_THRESHOLDS = {
    "minimum_mapping_unit_px": params.MIN_BLOB_PIXELS,
    "minimum_confidence": params.MIN_STORED_CONFIDENCE,
}
SORTS = ("confidence", "area", "date")
DISPLAY_WINDOW_FACTOR = 4.0  # the before/after view shows 4x the box's width and height (1.5 boxes of context per side)


@router.get("/changes", response_model=CandidateList)
def list_changes(
    job_id: Optional[int] = Query(default=None, description="Only this pair's candidates"),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    types: Optional[str] = Query(default=None, description="Comma-separated change types to include"),
    directions: Optional[str] = Query(default=None, description="Comma-separated directions to include"),
    sort: Literal["confidence", "area", "date"] = "confidence",
    hide_seasonal: bool = Query(default=False, description="Leave out candidates the persistence filter calls seasonal"),
    limit: int = Query(default=params.DISPLAY_CAP_PER_JOB, ge=1, le=1000, description="Cap per pair"),
) -> CandidateList:
    """Change candidates with the analyst's filters applied, plus the list of analysed pairs.

    The cap is per pair and keeps the strongest candidates by the chosen `sort`: the highest-confidence ones, or with
    sort=area the largest ones (so the biggest changes are never cut for having a modest confidence).
    """
    wanted = None
    if types:
        wanted = [t for t in (x.strip() for x in types.split(",")) if t]
        unknown = [t for t in wanted if t not in CHANGE_TYPES]
        if unknown:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown change type: {unknown[0]}")
    wanted_directions = None
    if directions:
        wanted_directions = [d for d in (x.strip() for x in directions.split(",")) if d]
        unknown = [d for d in wanted_directions if d not in DIRECTIONS]
        if unknown:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown direction: {unknown[0]}")

    conn = init_connection()
    try:
        init_schema(conn)
        all_jobs = store.list_jobs(conn)
        dates = dict(conn.execute("SELECT scene_id, acquisition_date FROM scenes").fetchall())
        scope = [j for j in all_jobs if job_id is None or j["job_id"] == job_id]
        scope_ids = [j["job_id"] for j in scope]
        stored = store.query_candidates(conn, job_ids=scope_ids)
        after_types = (
            store.query_candidates(conn, job_ids=scope_ids, types=wanted, directions=wanted_directions)
            if wanted or wanted_directions
            else stored
        )
        if hide_seasonal:
            after_types = [r for r in after_types if r["seasonality_status"] != "seasonal"]
        rows = [r for r in after_types if r["confidence"] >= min_confidence]
        area = analysed_area(conn, [j for j in scope if j["status"] == store.JOB_COMPLETED])
        per_job_stored: Dict[int, int] = {}
        for r in stored:
            per_job_stored[r["job_id"]] = per_job_stored.get(r["job_id"], 0) + 1
    finally:
        conn.close()

    def scoring(j: Dict[str, Any]) -> Dict[str, Any]:
        return (j["details"] or {}).get("scoring", {})

    found = sum(scoring(j).get("candidates_found", per_job_stored.get(j["job_id"], 0)) for j in scope)
    dropped_at_storage = sum(scoring(j).get("dropped_low_confidence", 0) for j in scope)
    excluded = (len(after_types) - len(rows)) + dropped_at_storage

    # Per-pair cap on the strongest candidates (rows arrive strongest first), then the requested display order
    taken: List[Dict[str, Any]] = []
    counts: Dict[int, int] = {}
    ranked = sorted(rows, key=lambda r: (-(r["area_px"] or 0), -r["confidence"])) if sort == "area" else rows
    for r in ranked:
        if counts.get(r["job_id"], 0) < limit:
            counts[r["job_id"]] = counts.get(r["job_id"], 0) + 1
            taken.append(r)
    if sort == "area":
        taken.sort(key=lambda r: (-(r["area_px"] or 0), -r["confidence"]))
    elif sort == "date":
        taken.sort(key=lambda r: (r["scene_b_date"] or "", r["confidence"]), reverse=True)

    pairs = [
        PairInfo(
            job_id=j["job_id"],
            scene_a_id=j["scene_a_id"],
            scene_b_id=j["scene_b_id"],
            scene_a_date=dates.get(j["scene_a_id"]),
            scene_b_date=dates.get(j["scene_b_id"]),
            status=j["status"],
            candidates=per_job_stored.get(j["job_id"], 0),
            found=scoring(j).get("candidates_found", per_job_stored.get(j["job_id"], 0)),
            valid_coverage=(j["details"] or {}).get("mutual_coverage_after_alignment", (j["details"] or {}).get("mutual_coverage")),
        )
        for j in sorted(all_jobs, key=lambda j: (dates.get(j["scene_b_id"]) or "", dates.get(j["scene_a_id"]) or ""), reverse=True)
    ]
    return CandidateList(
        total=len(stored),
        found=found,
        matching=len(rows),
        shown=len(taken),
        display_cap=limit,
        excluded_by_confidence=excluded,
        filtered_by_confidence=excluded > 0,
        candidates=[_summary(r) for r in taken],
        pairs=pairs,
        review_counts={k: sum(1 for r in stored if r["review_status"] == k) for k in ("pending", "confirmed", "rejected")},
        thresholds=DETECTION_THRESHOLDS,
        area=AnalysedArea(**area),
    )


@router.get("/changes/jobs", response_model=JobList)
def list_change_jobs() -> JobList:
    """Change-detection pair jobs and their outcome, so 'no candidates' can be told apart from 'not analysed'."""
    conn = init_connection()
    try:
        init_schema(conn)
        job_rows = store.list_jobs(conn)
        counts = dict(conn.execute("SELECT job_id, COUNT(*) FROM change_candidates GROUP BY job_id").fetchall())
        dates = dict(conn.execute("SELECT scene_id, acquisition_date FROM scenes").fetchall())
    finally:
        conn.close()
    return JobList(
        jobs=[
            JobSummary(
                job_id=j["job_id"],
                scene_a_id=j["scene_a_id"],
                scene_b_id=j["scene_b_id"],
                scene_a_date=dates.get(j["scene_a_id"]),
                scene_b_date=dates.get(j["scene_b_id"]),
                status=j["status"],
                phase=j["details"].get("phase"),
                error=j["error_message"],
                candidates=int(counts.get(j["job_id"], 0)),
            )
            for j in job_rows
        ]
    )


@router.get("/changes/{candidate_id}", response_model=CandidateDetail)
def get_change(candidate_id: int) -> CandidateDetail:
    """One candidate with both scenes, the confidence breakdown and the processing trace."""
    conn = init_connection()
    try:
        init_schema(conn)
        cand = store.get_candidate(conn, candidate_id)
        if cand is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown change candidate: {candidate_id}")
        scene_a = _scene_ref(conn, cand["scene_a_id"])
        scene_b = _scene_ref(conn, cand["scene_b_id"])
        reviews = store.get_reviews(conn, candidate_id)
        job_details = next((j["details"] for j in store.list_jobs(conn) if j["job_id"] == cand["job_id"]), {})
    finally:
        conn.close()

    w = params.CONFIDENCE_WEIGHTS
    values = {
        "alignment_quality": cand["alignment_quality"] or 0.0,
        "cluster_distance": cand["norm_cluster_dist"] or 0.0,
        "terrain_flatness": cand["terrain_flatness"] or 0.0,
        "valid_coverage": cand["valid_coverage"] or 0.0,
    }
    breakdown = {
        name: ConfidenceTerm(value=v, weight=w[name], contribution=v * w[name]) for name, v in values.items()
    }
    box = [cand["min_lon"], cand["min_lat"], cand["max_lon"], cand["max_lat"]]
    # The window opens on the ground both scenes cover; if only one extent is known, that one
    extent = intersect(scene_a.bounds_wgs84, scene_b.bounds_wgs84) or scene_b.bounds_wgs84 or scene_a.bounds_wgs84
    return CandidateDetail(
        **_summary(cand).model_dump(),
        display_bounds=display_bounds(box, extent, DISPLAY_WINDOW_FACTOR),
        processing_details=[TraceLine(**line) for line in trace_lines(job_details, cand)],
        bounds_native=[cand["min_x"], cand["min_y"], cand["max_x"], cand["max_y"]],
        scene_a=scene_a,
        scene_b=scene_b,
        confidence_breakdown=breakdown,
        confidence_factor=float(((cand.get("direction_evidence") or {}).get("seasonality") or {}).get("confidence_factor", 1.0)),
        direction_evidence=cand.get("direction_evidence"),
        seasonality=(cand.get("direction_evidence") or {}).get("seasonality"),
        decision_trace=job_details,
        reviews=[ReviewEntry(**r) for r in reviews],
    )


@router.post("/changes/{candidate_id}/review", response_model=CandidateSummary)
def review_change(candidate_id: int, payload: ReviewRequest) -> CandidateSummary:
    """Record an analyst's confirm/reject decision. Earlier decisions stay in the audit trail; the latest wins."""
    conn = init_connection()
    try:
        init_schema(conn)
        if store.get_candidate(conn, candidate_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown change candidate: {candidate_id}")
        try:
            analyst = getpass.getuser()  # single-analyst local tool: the OS user is the operator identifier
        except Exception:
            analyst = "local-operator"
        notes = payload.notes.strip() if payload.notes and payload.notes.strip() else None
        store.add_review(conn, candidate_id, payload.decision, analyst, notes)
        updated = store.get_candidate(conn, candidate_id)
    finally:
        conn.close()
    assert updated is not None
    return _summary(updated)
