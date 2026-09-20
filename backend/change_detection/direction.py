"""IRIS change detection — Phase 4b: direction classification (appearance / disappearance / expansion / contraction).

K-means clusters the *magnitude* of the difference, so its output has no sign: a binary mask cannot say whether
something appeared or vanished. Direction is rebuilt from what each date looks like on its own.

  Step 1  Classification Components: every pixel of each date is classified independently (vegetation / water /
          bare-or-built / other) from the analysis bands. They are never differenced and never touched by the
          morphology that cleans the Change Blobs.
  Step 2  For each surviving Change Blob, count the classes inside its footprint in date A and in date B, take the
          dominant class of each, and apply the direction rules below.
  Step 3  Refine the change type from the direction plus the signed NDVI.

Deliberate limits, recorded rather than hidden:
  * The analysis COG has B02/B03/B04/B08 only. No SWIR means no NDBI, so "built-up" cannot be told from bare soil
    (a documented failure of NDBI itself). The classes are therefore "bare/built", and "construction" is a best guess.
  * "Bare with higher NDBI" is approximated by a SWIR-free stand-in: visible brightness up by BUILT_BRIGHTNESS_GAIN
    together with a falling NDVI. It is a proxy to be calibrated, not a measurement of built-up-ness.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from change_detection import params
from change_detection.rasters import strips

logger = logging.getLogger("iris.change.direction")

# Classification Component classes (uint8 codes)
OTHER, VEGETATION, WATER, BARE = 0, 1, 2, 3
CLASS_NAMES = {OTHER: "other", VEGETATION: "vegetation", WATER: "water", BARE: "bare"}
N_CLASSES = 4

APPEARANCE = "appearance"
DISAPPEARANCE = "disappearance"
EXPANSION = "expansion"
CONTRACTION = "contraction"
UNCLASSIFIED = "unclassified"
DIRECTIONS = (APPEARANCE, DISAPPEARANCE, EXPANSION, CONTRACTION, UNCLASSIFIED)

# What a change is called once its direction and signed NDVI are known
CHANGE_TYPES = (
    "clearance",
    "construction",
    "revegetation",
    "urban_expansion",
    "water_loss",
    "vegetation_retreat",
    UNCLASSIFIED,
)


def to_reflectance(strip: np.ndarray, offsets: List[float], scale: float) -> np.ndarray:
    """(bands, rows, W) digital numbers -> reflectance, after the product's additive offset."""
    off = np.asarray(offsets, dtype=np.float32)[:, None, None]
    return (strip + off) / np.float32(scale)


def classify_strip(refl: np.ndarray) -> np.ndarray:
    """Per-pixel Classification Components for a (4, rows, W) reflectance strip ordered B02, B03, B04, B08.

      vegetation : NDVI > NDVI_VEG_MIN,  NDVI = (B08 - B04) / (B08 + B04)
      water      : NDWI > NDWI_WATER_MIN, NDWI = (B03 - B08) / (B03 + B08)   (green/NIR, both 10 m)
      bare/built : NDVI < NDVI_BARE_MAX and not water
      other      : everything else
    Water wins over the others: a pixel that reflects more green than NIR is not vegetation or soil.
    """
    green, red, nir = refl[1], refl[2], refl[3]
    with np.errstate(invalid="ignore", divide="ignore"):
        nir_red = nir + red
        green_nir = green + nir
        ndvi = np.where(nir_red > 1e-6, (nir - red) / nir_red, np.nan)
        ndwi = np.where(green_nir > 1e-6, (green - nir) / green_nir, np.nan)
    cls = np.full(ndvi.shape, OTHER, dtype=np.uint8)
    water = ndwi > params.NDWI_WATER_MIN
    cls[ndvi > params.NDVI_VEG_MIN] = VEGETATION
    cls[(ndvi < params.NDVI_BARE_MAX) & ~water] = BARE
    cls[water] = WATER
    return cls


def visible_brightness(refl: np.ndarray) -> np.ndarray:
    """Mean of the visible bands (B02, B03, B04): a SWIR-free brightness."""
    return refl[:3].mean(axis=0)


def _dominant(counts: np.ndarray) -> int:
    """Most common class; ties go to the more informative class (vegetation, water, bare, then other)."""
    best = int(np.max(counts))
    for cls in (VEGETATION, WATER, BARE, OTHER):
        if counts[cls] == best:
            return cls
    return OTHER


def decide(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    mean_dndvi: Optional[float],
    brightness_gain: Optional[float],
) -> Tuple[str, str, Dict[str, Any]]:
    """Direction and refined change type for one blob. Pure function of the class counts and two signed means.

    counts_a / counts_b: pixels of each class inside the blob footprint at date A and date B.
    mean_dndvi: mean NDVI(B) - NDVI(A) over the blob. brightness_gain: relative change in visible brightness.
    """
    a, b = _dominant(counts_a), _dominant(counts_b)
    area_a, area_b = int(counts_a[a]), int(counts_b[b])
    dndvi = mean_dndvi if mean_dndvi is not None and np.isfinite(mean_dndvi) else None
    dropped = dndvi is not None and dndvi <= -params.NDVI_DIRECTION_MIN
    rose = dndvi is not None and dndvi >= params.NDVI_DIRECTION_MIN
    built_proxy = (
        a == BARE
        and b == BARE
        and dropped
        and brightness_gain is not None
        and np.isfinite(brightness_gain)
        and brightness_gain >= params.BUILT_BRIGHTNESS_GAIN
    )

    # Direction, first matching rule wins
    if a in (VEGETATION, WATER) and b == BARE:
        direction, rule = DISAPPEARANCE, f"{CLASS_NAMES[a]} became bare"
    elif a == BARE and (b == VEGETATION or built_proxy):
        direction = APPEARANCE
        rule = "vegetation appeared on bare ground" if b == VEGETATION else "bare ground became brighter with falling NDVI (SWIR-free built-up proxy)"
    elif a == b and area_a > 0 and area_b > area_a * params.DIRECTION_AREA_GROWTH:
        direction, rule = EXPANSION, f"{CLASS_NAMES[a]} covers {area_b / max(area_a, 1):.1f}x more of the blob"
    elif a == b and area_a > 0 and area_b < area_a * params.DIRECTION_AREA_SHRINK:
        direction, rule = CONTRACTION, f"{CLASS_NAMES[a]} covers {area_b / max(area_a, 1):.1f}x of what it did"
    else:
        direction, rule = UNCLASSIFIED, "no direction rule matched"

    # Change type from direction + signed NDVI
    change_type = UNCLASSIFIED
    if direction == DISAPPEARANCE:
        if a == WATER and counts_b[WATER] < counts_a[WATER]:
            change_type = "water_loss"
        elif a == VEGETATION and dropped:
            change_type = "clearance"
    elif direction == APPEARANCE:
        if b == VEGETATION and rose:
            change_type = "revegetation"
        elif built_proxy:
            change_type = "construction"  # best guess without SWIR
    elif direction == EXPANSION:
        if a == BARE:
            change_type = "urban_expansion"
    elif direction == CONTRACTION:
        if a == VEGETATION:
            change_type = "vegetation_retreat"

    total_a, total_b = max(int(counts_a.sum()), 1), max(int(counts_b.sum()), 1)
    evidence = {
        "dominant_a": CLASS_NAMES[a],
        "dominant_b": CLASS_NAMES[b],
        "share_a": {CLASS_NAMES[i]: round(float(counts_a[i]) / total_a, 4) for i in range(N_CLASSES)},
        "share_b": {CLASS_NAMES[i]: round(float(counts_b[i]) / total_b, 4) for i in range(N_CLASSES)},
        "delta_ndvi": None if dndvi is None else round(float(dndvi), 4),
        "brightness_gain": None if brightness_gain is None or not np.isfinite(brightness_gain) else round(float(brightness_gain), 4),
        "rule": rule,
        "swir_available": False,
    }
    return direction, change_type, evidence


def classify_candidates(det: Any, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fill `direction`, `change_type` and `direction_evidence` on every candidate, in place.

    Reads the per-date Classification Components and brightness difference that the Phase 4 pass stored, plus the
    Change Blob labels. Each candidate carries the label of its blob in `blob_label`, which is removed here.
    """
    trace: Dict[str, Any] = {"available": det.class_a is not None, "by_direction": {}, "by_type": {}}
    for cand in candidates:
        cand.setdefault("direction", None)
        cand.setdefault("direction_evidence", None)

    if det.class_a is None or det.class_b is None or det.labels is None or not candidates:
        # No NIR band (or nothing to classify): a direction cannot be derived, so none is claimed
        for cand in candidates:
            cand.pop("blob_label", None)
        return trace

    labels = det.labels
    n = int(labels.max()) if labels.size else 0
    blob_labels = np.array([c["blob_label"] for c in candidates], dtype=np.int64)
    remap = np.full(n + 1, -1, dtype=np.int64)
    remap[blob_labels] = np.arange(len(candidates))

    m = len(candidates)
    counts_a = np.zeros(m * N_CLASSES, dtype=np.int64)
    counts_b = np.zeros(m * N_CLASSES, dtype=np.int64)
    bright_sum = np.zeros(m)
    bright_cnt = np.zeros(m)
    for row0, rows in strips(labels.shape[0]):
        lab = labels[row0 : row0 + rows]
        idx = remap[lab]
        sel = idx >= 0
        if not sel.any():
            continue
        i = idx[sel]
        ca = np.asarray(det.class_a[row0 : row0 + rows])[sel].astype(np.int64)
        cb = np.asarray(det.class_b[row0 : row0 + rows])[sel].astype(np.int64)
        counts_a += np.bincount(i * N_CLASSES + ca, minlength=m * N_CLASSES)
        counts_b += np.bincount(i * N_CLASSES + cb, minlength=m * N_CLASSES)
        db = np.asarray(det.dbright[row0 : row0 + rows])[sel]
        ok = np.isfinite(db)
        bright_sum += np.bincount(i[ok], weights=db[ok].astype(np.float64), minlength=m)
        bright_cnt += np.bincount(i[ok], minlength=m)
    counts_a = counts_a.reshape(m, N_CLASSES)
    counts_b = counts_b.reshape(m, N_CLASSES)

    for k, cand in enumerate(candidates):
        gain = float(bright_sum[k] / bright_cnt[k]) if bright_cnt[k] > 0 else None
        direction, change_type, evidence = decide(counts_a[k], counts_b[k], cand.get("mean_dndvi"), gain)
        cand["direction"] = direction
        cand["change_type"] = change_type
        cand["direction_evidence"] = evidence
        cand.pop("blob_label", None)
        trace["by_direction"][direction] = trace["by_direction"].get(direction, 0) + 1
        trace["by_type"][change_type] = trace["by_type"].get(change_type, 0) + 1
    logger.info("Direction: %s | types: %s", trace["by_direction"], trace["by_type"])
    return trace


# ---- search: what the analyst's words say about direction ---------------------------------------------------------

DIRECTION_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    APPEARANCE: ("new", "appeared", "built", "construction"),
    DISAPPEARANCE: ("cleared", "removed", "demolished", "deforestation"),
    EXPANSION: ("growing", "expanding", "sprawl"),
    CONTRACTION: ("shrinking", "retreating", "receding"),
}


def direction_hints(query: str) -> List[str]:
    """Directions whose keywords appear in the query. Whole-word matching on purpose: this is not NLP, and
    "renewal" must not read as "new"."""
    words = set(_words(query))
    return [d for d, keys in DIRECTION_KEYWORDS.items() if words.intersection(keys)]


def _words(text: str) -> List[str]:
    out, cur = [], []
    for ch in text.lower():
        if ch.isalpha():
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out
