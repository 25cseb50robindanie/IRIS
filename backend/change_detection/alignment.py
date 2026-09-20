"""IRIS change detection — Phase 2: grid check and geometric alignment.

Order is load-bearing: the grid check runs first, because a mutual validity mask is only meaningful once both
rasters are on the same pixel grid (architecture.md, Phase 2).
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from change_detection import params
from change_detection.rasters import (
    SceneReader,
    channel,
    matrix_from_full,
    matrix_to_full,
    max_corner_shift,
    strips,
)

logger = logging.getLogger("iris.change.alignment")


class AlignmentFailed(Exception):
    """Neither ECC nor the ORB fallback produced an alignment that passes its quality gate."""


class _Reject(Exception):
    """An alignment attempt finished but failed a quality gate (internal; triggers the next fallback)."""


@dataclass
class AlignmentResult:
    method: str  # "ecc" or "orb"
    matrix: Optional[np.ndarray]  # 3x3, A -> B pixel coordinates at full resolution; None when negligible
    quality: float  # ECC correlation coefficient, or ORB inlier ratio when ECC failed
    rho: Optional[float]
    inlier_ratio: Optional[float]
    max_shift_px: float
    decimation: int  # factor by which the images were decimated for estimation
    ecc_failure: Optional[str]  # why ECC was rejected, when the ORB fallback ran
    note: Optional[str] = None  # why estimation was skipped ("same_tile"), for the decision trace


def same_tile_alignment(tile_id: str) -> AlignmentResult:
    """Alignment for two scenes of one MGRS tile whose grids were verified identical.

    Sentinel-2 products of a tile share one fixed pixel grid, so there is nothing to estimate. Running an
    intensity-based method here does harm: ECC reads real land-cover change as misregistration and rejects
    good pairs. `quality` is a fixed 0.90: the grid is identical by construction, which is good but not measured,
    so it must not outrank an ECC correlation that was.
    """
    return AlignmentResult(
        method="same_tile",
        matrix=None,
        quality=params.SAME_TILE_ALIGNMENT_QUALITY,
        rho=None,
        inlier_ratio=None,
        max_shift_px=0.0,
        decimation=1,
        ecc_failure=None,
        note=f"Same MGRS tile {tile_id}; identical pixel grid verified, intensity-based alignment skipped",
    )


def _cv_reason(err: cv2.error) -> str:
    """A short, path-free description of an OpenCV error (its message embeds the build machine's file paths)."""
    m = re.search(r"error: \((-?\d+):([^)]*)\)\s*(.*?)(?:\s+in function|$)", str(err), re.S)
    if not m:
        # Not OpenCV's usual format: keep the wording, drop anything that looks like a path
        text = re.sub(r"\S*[\\/]\S*", "<path>", " ".join(str(err).split()))
        return text[:160] or "OpenCV error"
    detail = " ".join(m.group(3).split())
    return f"{m.group(2).strip()} ({detail[:140]})" if detail else m.group(2).strip()


def regrid_to_reference(src_path: Path, ref_path: Path, dst_path: Path, resampling: Resampling) -> None:
    """Warp `src_path` onto the pixel grid of `ref_path` (band by band, out of core via GDAL).

    Imagery uses bilinear; the categorical SCL uses nearest-neighbour. Pixels outside the source footprint are
    left as no-data (0), which the validity mask then treats as invalid.
    """
    with rasterio.open(str(ref_path)) as ref, rasterio.open(str(src_path)) as src:
        nodata = src.nodata if src.nodata is not None else 0
        profile = {
            "driver": "GTiff",
            "width": ref.width,
            "height": ref.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": ref.crs,
            "transform": ref.transform,
            "nodata": nodata,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": "DEFLATE",
        }
        with rasterio.open(str(dst_path), "w", **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    resampling=resampling,
                    src_nodata=nodata,
                    dst_nodata=nodata,
                )


def _decimated_channel(reader: SceneReader, factor: int) -> np.ndarray:
    """Block-mean of the differenced channel, decimated by `factor` (float32, shape H//f x W//f)."""
    hd, wd = reader.height // factor, reader.width // factor
    out = np.zeros((hd, wd), dtype=np.float32)
    for row0, rows in strips(reader.height):
        use = (rows // factor) * factor  # STRIP_ROWS is a multiple of every factor used, so row0 is aligned
        if use == 0:
            continue
        ch = channel(reader._read(row0, row0 + use))[:, : wd * factor]
        out[row0 // factor : row0 // factor + use // factor] = ch.reshape(use // factor, factor, wd, factor).mean(axis=(1, 3))
    return out


def _decimated_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """A decimated pixel is valid only if every full-resolution pixel in its block is valid."""
    hd, wd = mask.shape[0] // factor, mask.shape[1] // factor
    return mask[: hd * factor, : wd * factor].reshape(hd, factor, wd, factor).all(axis=(1, 3))


def _standardise(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero-mean/unit-variance over valid pixels; invalid pixels become NaN and are then filled with 0.

    Filling with 0 (the valid-pixel mean after standardising) is required as well as the input mask: ECC
    raises "NaN encountered" on a single NaN, and NaN left in the buffer propagates through the projection
    arithmetic regardless of the mask.
    """
    vals = img[mask]
    sd = float(vals.std())
    if sd < 1e-6:
        raise _Reject("scene has no texture on the valid area")
    out = np.where(mask, (img - float(vals.mean())) / sd, np.nan).astype(np.float32)
    return np.nan_to_num(out, nan=0.0)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(128.0 + img * 42.0, 0, 255).astype(np.uint8)


def _block_mean2(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    return img[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)).astype(np.float32)


def _ecc(a: np.ndarray, b: np.ndarray, m: np.ndarray, factor: int, init_full: Optional[np.ndarray]) -> Tuple[np.ndarray, float]:
    """One ECC run at decimation `factor`. Returns (3x3 full-resolution A->B matrix, rho)."""
    init = np.eye(2, 3, dtype=np.float32)
    if init_full is not None:
        init = matrix_from_full(init_full, factor)[:2].astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, params.ECC_MAX_ITERATIONS, params.ECC_EPSILON)
    # MOTION_EUCLIDEAN: scenes share a grid, so the residual is a small shift/rotation; homography's extra
    # parameters diverge more readily. The mask is the mutual validity mask; gaussFiltSize is passed explicitly.
    rho, warp = cv2.findTransformECC(
        np.ascontiguousarray(a),
        np.ascontiguousarray(b),
        init,
        cv2.MOTION_EUCLIDEAN,
        criteria,
        m.astype(np.uint8),
        params.ECC_GAUSS_FILT_SIZE,
    )
    m3 = np.vstack([warp.astype(np.float64), [0.0, 0.0, 1.0]])
    return matrix_to_full(m3, factor), float(rho)


def _ecc_pyramid(a: np.ndarray, b: np.ndarray, m: np.ndarray, factor: int) -> Tuple[np.ndarray, float]:
    """Coarse-to-fine ECC: a large initial displacement is found at half resolution, then refined."""
    init_full: Optional[np.ndarray] = None
    if min(a.shape) >= 256:
        ca, cb = _block_mean2(a), _block_mean2(b)
        cm = _decimated_mask(m, 2)
        ca, cb = ca[: cm.shape[0], : cm.shape[1]], cb[: cm.shape[0], : cm.shape[1]]
        try:
            init_full, _ = _ecc(ca, cb, cm, factor * 2, None)
        except cv2.error:
            # The coarse level is only an initialiser; the fine level still gets its own attempt from identity
            init_full = None
    return _ecc(a, b, m, factor, init_full)


def _orb(a: np.ndarray, b: np.ndarray, m: np.ndarray, factor: int) -> Tuple[np.ndarray, float]:
    """ORB + BFMatcher + RANSAC homography. Returns (3x3 full-resolution A->B matrix, inlier ratio)."""
    mask255 = m.astype(np.uint8) * 255
    orb = cv2.ORB_create(nfeatures=params.ORB_FEATURES)
    kp_a, des_a = orb.detectAndCompute(_to_uint8(a), mask255)
    kp_b, des_b = orb.detectAndCompute(_to_uint8(b), mask255)
    if des_a is None or des_b is None:
        raise _Reject("ORB found no keypoints")
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des_a, des_b)
    if len(matches) < params.ORB_MIN_MATCHES:
        raise _Reject(f"only {len(matches)} ORB matches (need {params.ORB_MIN_MATCHES})")

    pts_a = np.float32([kp_a[x.queryIdx].pt for x in matches])
    pts_b = np.float32([kp_b[x.trainIdx].pt for x in matches])
    hom, inliers = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, params.ORB_RANSAC_REPROJ_THRESHOLD)
    if hom is None or inliers is None or not np.isfinite(hom).all():
        raise _Reject("RANSAC found no homography")
    ratio = float(inliers.sum()) / len(matches)
    if ratio < params.ORB_MIN_INLIER_RATIO:
        raise _Reject(f"inlier ratio {ratio:.2f} below {params.ORB_MIN_INLIER_RATIO}")
    return matrix_to_full(hom.astype(np.float64), factor), ratio


def estimate_alignment(reader_a: SceneReader, reader_b: SceneReader, mutual: np.ndarray) -> AlignmentResult:
    """Estimate the transform taking scene A's pixels to scene B's, on the NIR band over the mutual validity mask.

    ECC is primary. It is rejected on any cv2.error (both the "NaN encountered" and the "stopped before
    convergence" failures are raised, not returned), on a correlation below ECC_RHO_MIN (ECC can converge to a
    bad optimum with no exception at all), or on an implausibly large shift. Rejection falls back to
    ORB + RANSAC; if that fails its own gate too, the pair is rejected with AlignmentFailed.
    """
    height, width = reader_a.height, reader_a.width
    factor = 1
    while (height // factor) * (width // factor) > params.ECC_MAX_PIXELS:
        factor *= 2

    mask_d = _decimated_mask(mutual, factor)
    if int(mask_d.sum()) < 100:
        raise AlignmentFailed("too little valid area to estimate an alignment")

    try:
        a = _standardise(_decimated_channel(reader_a, factor), mask_d)
        b = _standardise(_decimated_channel(reader_b, factor), mask_d)
    except _Reject as e:
        raise AlignmentFailed(str(e)) from e

    ecc_failure: Optional[str] = None
    try:
        m_full, rho = _ecc_pyramid(a, b, mask_d, factor)
        shift = max_corner_shift(m_full, width, height)
        if not np.isfinite(rho) or rho < params.ECC_RHO_MIN:
            raise _Reject(f"rho {rho:.3f} below {params.ECC_RHO_MIN}")
        if shift > params.MAX_ALIGNMENT_SHIFT_PX:
            raise _Reject(f"implausible shift of {shift:.1f}px")
        logger.info("ECC alignment accepted: rho=%.4f shift=%.2fpx (decimation %d)", rho, shift, factor)
        return AlignmentResult(
            "ecc", None if shift < params.IDENTITY_SHIFT_PX else m_full, min(max(rho, 0.0), 1.0), rho, None, shift, factor, None
        )
    except cv2.error as e:
        ecc_failure = f"cv2.error: {_cv_reason(e)}"
    except _Reject as e:
        ecc_failure = str(e)
    logger.warning("ECC rejected (%s); falling back to ORB + RANSAC", ecc_failure)

    try:
        m_full, ratio = _orb(a, b, mask_d, factor)
        shift = max_corner_shift(m_full, width, height)
        if shift > params.MAX_ALIGNMENT_SHIFT_PX:
            raise _Reject(f"implausible shift of {shift:.1f}px")
    except (_Reject, cv2.error) as e:
        reason = _cv_reason(e) if isinstance(e, cv2.error) else str(e)
        raise AlignmentFailed(f"ECC rejected ({ecc_failure}); ORB fallback also failed ({reason})") from e

    logger.info("ORB alignment accepted: inlier_ratio=%.3f shift=%.2fpx", ratio, shift)
    return AlignmentResult(
        "orb", None if shift < params.IDENTITY_SHIFT_PX else m_full, min(max(ratio, 0.0), 1.0), None, ratio, shift, factor, ecc_failure
    )


def warp_mask(mask: np.ndarray, matrix: Optional[np.ndarray]) -> np.ndarray:
    """Carry a boolean validity mask through the alignment transform with nearest-neighbour (it is categorical)."""
    if matrix is None:
        return mask
    from change_detection.rasters import warp_rows

    height, width = mask.shape
    as_u8 = mask.astype(np.uint8)
    out = np.zeros_like(mask)
    for row0, rows in strips(height):
        strip = warp_rows(
            lambda lo, hi: as_u8[None, lo:hi], matrix, row0, rows, width, height, interpolation=cv2.INTER_NEAREST, dtype=np.uint8
        )
        out[row0 : row0 + rows] = strip[0] > 0
    return out


def describe(result: AlignmentResult) -> dict:
    """JSON-safe summary for the decision trace."""
    return {
        "method": result.method,
        "rho": None if result.rho is None else round(result.rho, 5),
        "inlier_ratio": None if result.inlier_ratio is None else round(result.inlier_ratio, 4),
        "quality": round(result.quality, 5),
        "max_shift_px": round(result.max_shift_px, 3),
        "warp_applied": result.matrix is not None,
        "decimation": result.decimation,
        "ecc_failure": result.ecc_failure,
        "note": result.note,
        "matrix": None if result.matrix is None else [[round(float(v), 6) for v in row] for row in result.matrix],
    }
