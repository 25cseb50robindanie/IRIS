"""IRIS change detection — Phase 4: difference image, K-means, morphology, connected components.

NaN contract (architecture.md, Phase 1): invalid pixels are carried as an explicit boolean mask. K-means only
ever sees the extracted valid pixels, and its labels are scattered back into a raster initialised to 0
("unlabelled"). Nothing is imputed to zero in the difference image.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

from change_detection import direction, params
from change_detection.radiometry import Normalization
from change_detection.rasters import SceneReader, channel, dn_scale, strips, uses_nir

logger = logging.getLogger("iris.change.detection")


@dataclass
class Detection:
    """Everything Phase 5 needs, plus the figures for the decision trace."""
    abs_diff: np.ndarray  # (H, W) float32 memmap, NaN outside the mutual valid area
    dndvi: Optional[np.ndarray]  # signed NDVI(B) - NDVI(A), or None without a NIR band
    labels: Optional[np.ndarray]  # (H, W) int32 connected components of the cleaned mask, or None if no change
    areas: Optional[np.ndarray]  # pixels per label (index 0 = background)
    keep: Optional[np.ndarray]  # bool per label: survives the minimum-blob-size filter
    objects: Optional[list]  # bounding-box slices from ndimage.find_objects (label i -> objects[i - 1])
    c_low: float  # no-change centroid
    c_high: float  # change centroid
    threshold: float
    trace: dict
    no_change_reason: Optional[str] = None
    # Phase 4b inputs, stored during the difference pass: per-date Classification Components and the relative
    # change in visible brightness (all None without a NIR band)
    class_a: Optional[np.ndarray] = None
    class_b: Optional[np.ndarray] = None
    dbright: Optional[np.ndarray] = None
    group_of: Optional[np.ndarray] = None  # detection group per blob label (grouping.assign_groups)
    group_large: Optional[np.ndarray] = None  # bool per blob label: merged by the large-area pass
    group_radius: Optional[np.ndarray] = None  # dilation (px) that produced each blob's group


def _parse_offsets(tags: dict, count: int) -> List[float]:
    raw = tags.get("BOA_ADD_OFFSETS", "")
    try:
        vals = [float(v) for v in raw.split(",") if v != ""]
    except ValueError:
        vals = []
    return (vals + [0.0] * count)[:count]


def _ndvi(strip: np.ndarray, offsets: List[float], scale: float) -> np.ndarray:
    """NDVI from bands 3 (red) and 4 (NIR), after removing the additive offset and applying the quantification."""
    red = (strip[2] + offsets[2]) / scale
    nir = (strip[3] + offsets[3]) / scale
    denom = nir + red
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denom > 1e-6, (nir - red) / denom, np.nan)
    return out.astype(np.float32)


def detect_changes(
    reader_a: SceneReader,
    reader_b: SceneReader,
    mutual: np.ndarray,
    norm: Normalization,
    work_dir: Path,
) -> Detection:
    height, width = reader_a.height, reader_a.width
    scale = dn_scale(reader_a.dtype)
    has_nir = uses_nir(reader_a.count)
    offsets = _parse_offsets(reader_a.tags, reader_a.count)
    valid_total = int(mutual.sum())

    work_dir.mkdir(parents=True, exist_ok=True)
    abs_diff = np.lib.format.open_memmap(str(work_dir / "abs_diff.npy"), mode="w+", dtype=np.float32, shape=(height, width))
    dndvi = (
        np.lib.format.open_memmap(str(work_dir / "dndvi.npy"), mode="w+", dtype=np.float32, shape=(height, width))
        if has_nir
        else None
    )

    class_a = class_b = dbright = None
    if has_nir:
        shape = (height, width)
        class_a = np.lib.format.open_memmap(str(work_dir / "class_a.npy"), mode="w+", dtype=np.uint8, shape=shape)
        class_b = np.lib.format.open_memmap(str(work_dir / "class_b.npy"), mode="w+", dtype=np.uint8, shape=shape)
        dbright = np.lib.format.open_memmap(str(work_dir / "dbright.npy"), mode="w+", dtype=np.float32, shape=shape)

    # Difference on the NIR band (B08): |B_normalised - A|. Also collect a bounded random sample of the valid
    # values for K-means, and (with a NIR band) the signed dNDVI used to name the change.
    rng = np.random.default_rng(params.KMEANS_SEED)
    rate = min(1.0, params.KMEANS_SAMPLE_MAX / max(valid_total, 1))
    samples = []
    for row0, rows in strips(height):
        m = mutual[row0 : row0 + rows]
        a = reader_a.strip(row0, rows)
        b = norm.apply(reader_b.strip(row0, rows))
        diff = np.abs(channel(b) - channel(a))
        diff[~m] = np.nan
        abs_diff[row0 : row0 + rows] = diff
        if dndvi is not None:
            dn = _ndvi(b, offsets, scale) - _ndvi(a, offsets, scale)
            dn[~m] = np.nan
            dndvi[row0 : row0 + rows] = dn
        if class_a is not None:
            # Classification Components: each date on its own, never differenced (B on A's radiometric baseline)
            refl_a = direction.to_reflectance(a, offsets, scale)
            refl_b = direction.to_reflectance(b, offsets, scale)
            class_a[row0 : row0 + rows] = direction.classify_strip(refl_a)
            class_b[row0 : row0 + rows] = direction.classify_strip(refl_b)
            vis_a, vis_b = direction.visible_brightness(refl_a), direction.visible_brightness(refl_b)
            with np.errstate(invalid="ignore", divide="ignore"):
                gain = (vis_b - vis_a) / np.maximum(vis_a, 0.01)
            gain[~m] = np.nan
            dbright[row0 : row0 + rows] = gain
        vals = diff[m]
        if rate < 1.0:
            vals = vals[rng.random(vals.size) < rate]
        samples.append(vals)
    sample = np.concatenate(samples) if samples else np.empty(0, dtype=np.float32)

    trace = {
        "difference_band": "B08 (NIR)" if has_nir else "mean intensity (no NIR band available)",
        "valid_pixels": valid_total,
        "kmeans_sample": int(sample.size),
    }

    def no_change(reason: str, c_low: float = 0.0, c_high: float = 0.0, thr: float = 0.0) -> Detection:
        logger.info("No change reported: %s", reason)
        trace["result"] = reason
        return Detection(
            abs_diff, dndvi, None, None, None, None, c_low, c_high, thr, trace, reason,
            class_a=class_a, class_b=class_b, dbright=dbright,
        )

    if sample.size < 2 or float(np.ptp(sample)) < 1e-9:
        return no_change("the difference image has no variation")

    # K-means (k=2) on the extracted valid values only
    km = KMeans(n_clusters=2, n_init=10, random_state=params.KMEANS_SEED).fit(sample.reshape(-1, 1).astype(np.float64))
    centers = km.cluster_centers_.ravel()
    # Cluster ids are arbitrary; the change cluster is the one with the HIGHER centroid, never "cluster 1"
    hi = int(np.argmax(centers))
    c_high, c_low = float(centers[hi]), float(centers[1 - hi])
    threshold = (c_high + c_low) / 2.0  # in 1-D, nearest-centroid assignment is a midpoint threshold
    min_magnitude = params.MIN_CHANGE_MAGNITUDE * scale
    trace.update(
        {
            "kmeans_centroids": [round(c_low, 4), round(c_high, 4)],
            "kmeans_threshold": round(threshold, 4),
            "min_change_magnitude": round(min_magnitude, 4),
        }
    )
    if c_high < min_magnitude:
        return no_change(
            f"change centroid {c_high:.1f} is below the minimum change magnitude {min_magnitude:.1f}", c_low, c_high, threshold
        )

    # Scatter labels back into a full raster; unlabelled (invalid) pixels stay 0
    change = np.zeros((height, width), dtype=np.uint8)
    for row0, rows in strips(height):
        change[row0 : row0 + rows] = np.asarray(abs_diff[row0 : row0 + rows]) > threshold  # NaN > x is False
    trace["raw_change_pixels"] = int(change.sum())

    # Morphological cleanup on the change blobs: opening (removes salt-and-pepper), then closing (fills gaps)
    kernel = np.ones((params.MORPH_KERNEL_SIZE, params.MORPH_KERNEL_SIZE), dtype=np.uint8)
    change = cv2.morphologyEx(change, cv2.MORPH_OPEN, kernel)
    change = cv2.morphologyEx(change, cv2.MORPH_CLOSE, kernel)
    change[~mutual] = 0  # closing can spill into invalid pixels; never claim change where nothing was observed
    trace["cleaned_change_pixels"] = int(change.sum())

    labels, n = ndimage.label(change)
    labels = labels.astype(np.int32, copy=False)
    del change
    areas = np.zeros(n + 1, dtype=np.int64)
    for row0, rows in strips(height):
        areas += np.bincount(labels[row0 : row0 + rows].ravel(), minlength=n + 1)
    keep = areas >= params.MIN_BLOB_PIXELS
    keep[0] = False
    trace.update({"components_total": int(n), "components_kept": int(keep.sum()), "min_blob_pixels": params.MIN_BLOB_PIXELS})
    logger.info("Components: %d found, %d kept (>= %d px)", n, int(keep.sum()), params.MIN_BLOB_PIXELS)

    objects = ndimage.find_objects(labels) if n else []
    return Detection(
        abs_diff, dndvi, labels, areas, keep, objects, c_low, c_high, threshold, trace,
        class_a=class_a, class_b=class_b, dbright=dbright,
    )
