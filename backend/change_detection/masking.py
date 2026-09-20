"""IRIS change detection — Phase 1: quality masking and trust scoring."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.windows import Window

from change_detection import params
from change_detection.rasters import strips

logger = logging.getLogger("iris.change.masking")


def _valid_lookup() -> np.ndarray:
    lut = np.zeros(256, dtype=bool)
    lut[list(params.SCL_VALID_CLASSES)] = True
    return lut


@dataclass
class Validity:
    """Validity mask of one scene plus its Phase 1 trust figures."""
    valid: np.ndarray  # (H, W) bool — the mask is carried as a first-class array, never inferred from NaN
    cloud_trust: float  # 1 - invalid / total pixels
    invalid_fraction: float
    snow_fraction: float  # valid but flagged: snow/ice
    source: str  # "scl" or "data_mask"


def build_validity(analysis_path: Path, scl_path: Optional[Path]) -> Validity:
    """Validity mask for one scene.

    With an SCL: classes 4/5/6/7/11 are valid, 0/1/2/3/8/9/10 invalid. Without one (non-Sentinel-2 input) every
    pixel that carries data is valid and CloudTrust is 1.0. Either way a pixel with no data in the analysis raster
    is invalid: an outside-the-footprint zero is not an observation.
    """
    lut = _valid_lookup()
    with rasterio.open(str(analysis_path)) as ref:
        height, width = ref.height, ref.width
        valid = np.zeros((height, width), dtype=bool)
        snow = 0
        scl_src = rasterio.open(str(scl_path)) if scl_path else None
        try:
            if scl_src is not None and (scl_src.height, scl_src.width) != (height, width):
                raise ValueError("SCL raster is not on the analysis grid")
            for row0, rows in strips(height):
                window = Window(0, row0, width, rows)
                data_ok = (ref.read_masks(window=window) > 0).all(axis=0)
                if scl_src is not None:
                    scl = scl_src.read(1, window=window)
                    strip_valid = lut[scl] & data_ok
                    snow += int(((scl == params.SCL_SNOW_CLASS) & data_ok).sum())
                else:
                    strip_valid = data_ok
                valid[row0 : row0 + rows] = strip_valid
        finally:
            if scl_src is not None:
                scl_src.close()

    total = height * width
    invalid_fraction = 1.0 - float(valid.sum()) / total if total else 1.0
    if scl_path is None:
        logger.warning("No SCL for %s: treating every data pixel as valid, CloudTrust=1.0", analysis_path.name)
        return Validity(valid, 1.0, invalid_fraction, 0.0, "data_mask")
    return Validity(valid, 1.0 - invalid_fraction, invalid_fraction, snow / total if total else 0.0, "scl")
