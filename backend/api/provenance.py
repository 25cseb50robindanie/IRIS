"""IRIS provenance helpers — how a change candidate came to exist, in words and in fields.

One place turns a job's decision trace (job.details) into (a) the `processing` block of an exported feature and (b) the
lines of the "Processing Details" panel, so what the analyst reads on screen and what leaves in the GeoJSON cannot drift.
"""

from typing import Any, Dict, List, Optional, Sequence

from change_detection import params
from mgrs_ref import bounds_centre_mgrs

NIR_INDEX = 3  # B08 in the analysis COG (B02, B03, B04, B08)


def _scene_mask(details: Dict[str, Any], which: str) -> Dict[str, Any]:
    return (details.get("masking") or {}).get(which) or {}


def _masking_source(details: Dict[str, Any]) -> str:
    a, b = _scene_mask(details, "scene_a").get("source"), _scene_mask(details, "scene_b").get("source")
    names = {"scl": "SCL", "qa_pixel": "QA_PIXEL", "heuristic": "heuristic", "data_mask": "data_mask"}
    if a and a == b:
        return names.get(a, a)
    if a or b:
        return f"{names.get(a, a)} (A) / {names.get(b, b)} (B)"
    return "unknown"


def _alignment_method(details: Dict[str, Any]) -> str:
    method = (details.get("alignment") or {}).get("method")
    return {"same_tile": "same_mgrs_grid"}.get(method, method or "unknown")


def _normalisation(details: Dict[str, Any]) -> str:
    tier = (details.get("radiometry") or {}).get("tier")
    return {"pif": "pif_robust_regression", "identity": "none"}.get(tier, tier or "unknown")


def processing_block(details: Dict[str, Any]) -> Dict[str, Any]:
    """The `processing` object of an exported feature."""
    a, b = _scene_mask(details, "scene_a"), _scene_mask(details, "scene_b")
    alignment = details.get("alignment") or {}
    return {
        "masking": _masking_source(details),
        "cloud_trust_a": a.get("cloud_trust"),
        "cloud_trust_b": b.get("cloud_trust"),
        "alignment": _alignment_method(details),
        "alignment_quality": alignment.get("quality"),
        "normalisation": _normalisation(details),
        "pif_pixels": (details.get("radiometry") or {}).get("pif_pixels"),
    }


def _fmt(v: Optional[float], digits: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def trace_lines(details: Dict[str, Any], cand: Dict[str, Any]) -> List[Dict[str, str]]:
    """The full decision trace of one candidate as labelled sentences, in pipeline order."""
    lines: List[Dict[str, str]] = []

    def add(key: str, label: str, text: str) -> None:
        lines.append({"key": key, "label": label, "text": text})

    # Masking
    a, b = _scene_mask(details, "scene_a"), _scene_mask(details, "scene_b")
    trust = f"CloudTrust A: {_fmt(a.get('cloud_trust'))}, CloudTrust B: {_fmt(b.get('cloud_trust'))}"
    src_a, src_b = a.get("source"), b.get("source")
    described = {
        "scl": "SCL-based",
        "qa_pixel": "QA_PIXEL-based",
        "heuristic": "Heuristic (no QA band: CloudTrust reduced, no pixel removed)",
    }
    if src_a and src_a == src_b:
        add("masking", "Masking", f"{described.get(src_a, 'Data-mask based (no QA band available)')}, {trust}")
    elif src_a or src_b:
        add("masking", "Masking", f"A: {src_a or 'unknown'}, B: {src_b or 'unknown'}, {trust}")

    # Alignment
    al = details.get("alignment") or {}
    method = al.get("method")
    if method == "same_tile":
        tiles = al.get("tiles") or []
        tile = next((t for t in tiles if t), None)
        if tile and not tile.startswith("T"):
            tile = f"T{tile}"  # Sentinel-2 names a tile T43PHM; the pipeline keeps the bare 43PHM
        add("alignment", "Alignment", f"Same MGRS tile ({tile or 'unknown'}) — grid verified by definition")
    elif method == "ecc":
        add("alignment", "Alignment", f"ECC registration, correlation {_fmt(al.get('rho'), 3)}, largest shift {_fmt(al.get('max_shift_px'))} px")
    elif method == "orb":
        add("alignment", "Alignment", f"ORB feature matching, inlier ratio {_fmt(al.get('inlier_ratio'))}, largest shift {_fmt(al.get('max_shift_px'))} px")
    elif method:
        add("alignment", "Alignment", str(method))

    # Normalisation (gain and offset of the NIR band, the one that is differenced)
    rad = details.get("radiometry") or {}
    if rad.get("tier") == "pif":
        slope, intercept = rad.get("slope") or [], rad.get("intercept") or []
        i = NIR_INDEX if len(slope) > NIR_INDEX else len(slope) - 1
        gain = f", gain: {_fmt(slope[i])}, offset: {_fmt(intercept[i], 1)}" if slope and intercept else ""
        add("normalisation", "Normalisation", f"PIF robust regression, {int(rad.get('pif_pixels') or 0):,} anchor pixels{gain}")
    elif rad:
        notes = "; ".join(rad.get("notes") or [])
        add("normalisation", "Normalisation", f"None applied (too few stable pixels){': ' + notes if notes else ''}")

    # Detection
    det = details.get("detection") or {}
    centroids = det.get("kmeans_centroids")
    if centroids:
        nir = str(det.get("difference_band", "")).startswith("B08")
        add(
            "detection",
            "Detection",
            f"{'NIR difference' if nir else 'Mean-intensity difference'} → K-Means K=2, change centroid magnitude: {centroids[1]:.1f}",
        )

    # Grouping, only when this detection is several blobs
    blobs = cand.get("sub_blobs") or 1
    if blobs > 1:
        trace = details.get("grouping") or {}
        merged = (cand.get("direction_evidence") or {}).get("merged") or {}
        large = bool(merged.get("large_area"))
        grow = merged.get("dilation_px") or (params.LARGE_AREA_DILATION_PX if large else trace.get("dilation_px", params.GROUP_DILATION_PX))
        kind = "Large-area change: merged" if large else "Merged"
        add(
            "grouping",
            "Grouping",
            f"{kind} from {blobs} change blobs less than {2 * grow} px apart; area, outline and ΔNDVI are those of the blobs themselves",
        )

    # Direction and evidence
    ev = cand.get("direction_evidence")
    direction = cand.get("direction") or "unclassified"
    if ev:
        text = f"{direction.capitalize()} — dominant class A: {ev.get('dominant_a')}, dominant class B: {ev.get('dominant_b')}"
        if direction == "unclassified" and ev.get("rule"):
            text += f" ({ev['rule']})"
        add("direction", "Direction", text)
    else:
        add("direction", "Direction", "Not classified (analysed before direction classification, or without a NIR band)")
    if cand.get("mean_dndvi") is not None:
        add("evidence", "Evidence", f"ΔNDVI: {cand['mean_dndvi']:.2f}")

    # Seasonal persistence filter, only for candidates it applied to
    status = cand.get("seasonality_status")
    if status:
        s = (ev or {}).get("seasonality") or {}
        clear = s.get("priors_clear", 0)
        if status == "unverified":
            text = (
                f"Unverified — {clear} clear same-season prior observation{'s' if clear != 1 else ''} "
                f"(needs {params.SEASONAL_MIN_PRIORS}); confidence ×{params.UNVERIFIED_CONFIDENCE_FACTOR}"
            )
        else:
            spread = (
                f"NDVI {_fmt(s.get('current_ndvi'), 2)} against a same-season mean of {_fmt(s.get('prior_mean_ndvi'), 2)} "
                f"(±{_fmt(s.get('prior_std_ndvi'), 2)}) over {clear} prior years"
            )
            if status == "seasonal":
                text = f"Seasonal — {spread}; within {params.SEASONAL_SIGMAS:g} std devs, confidence ×{params.SEASONAL_CONFIDENCE_FACTOR}"
            else:
                text = f"Anomalous — {spread}; drop exceeds {params.SEASONAL_SIGMAS:g} std devs, confidence unchanged"
        add("seasonality", "Seasonality", text)
    return lines


def display_bounds(
    bounds: Sequence[float], extent: Optional[Sequence[float]], factor: float = 4.0
) -> List[float]:
    """Window for the before/after view: `factor` times the box's width and height, centred on it.

    Clamped to `extent` (the area both scenes cover). Where the box sits near an edge the window slides inwards so it
    keeps its size, and only shrinks when the extent itself is smaller. Both are WGS84 [min_lon, min_lat, max_lon, max_lat].
    """
    min_x, min_y, max_x, max_y = (float(v) for v in bounds)
    cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    half_w, half_h = (max_x - min_x) * factor / 2.0, (max_y - min_y) * factor / 2.0
    lo_x, lo_y, hi_x, hi_y = cx - half_w, cy - half_h, cx + half_w, cy + half_h
    if extent:
        e_min_x, e_min_y, e_max_x, e_max_y = (float(v) for v in extent)
        if hi_x - lo_x >= e_max_x - e_min_x:
            lo_x, hi_x = e_min_x, e_max_x
        else:
            shift = max(0.0, e_min_x - lo_x) - max(0.0, hi_x - e_max_x)
            lo_x, hi_x = lo_x + shift, hi_x + shift
        if hi_y - lo_y >= e_max_y - e_min_y:
            lo_y, hi_y = e_min_y, e_max_y
        else:
            shift = max(0.0, e_min_y - lo_y) - max(0.0, hi_y - e_max_y)
            lo_y, hi_y = lo_y + shift, hi_y + shift
    return [lo_x, lo_y, hi_x, hi_y]


def intersect(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[List[float]]:
    """Overlap of two WGS84 boxes, or None when there is none (or one is unknown)."""
    if not a or not b:
        return None
    box = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    return box if box[0] < box[2] and box[1] < box[3] else None


def analysed_area(conn: Any, pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The ground a set of analysed pairs covers: the union of each pair's shared footprint, and its MGRS reference."""
    boxes: List[List[float]] = []
    for pair in pairs:
        footprints = []
        for scene_id in (pair["scene_a_id"], pair["scene_b_id"]):
            row = conn.execute(
                "SELECT f.min_lon, f.min_lat, f.max_lon, f.max_lat FROM scene_footprints f "
                "JOIN scenes s ON s.rowid = f.id WHERE s.scene_id = ?",
                (scene_id,),
            ).fetchone()
            footprints.append(list(row) if row else None)
        shared = intersect(footprints[0], footprints[1])
        if shared:
            boxes.append(shared)
    if not boxes:
        return {"bounds": None, "mgrs": None}
    bounds = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
    return {"bounds": [round(v, 6) for v in bounds], "mgrs": bounds_centre_mgrs(bounds)}

