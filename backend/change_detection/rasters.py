"""IRIS change detection — windowed raster access.

Nothing in the pipeline loads a whole scene: rasters are read, warped and reduced one full-width strip at a
time, so peak memory does not grow with scene height (architecture.md, "Raster I/O").
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

from change_detection import params


# An MGRS tile ID inside a Sentinel-2 product name: "T" + 2-digit UTM zone + latitude band + 2-letter square.
# Anchored to separators so a time such as "20220623T052701" can never match.
_MGRS_TILE = re.compile(r"(?:^|[_\-.])T(\d{2}[C-HJ-NP-X][A-Z]{2})(?=$|[_\-.])")


def mgrs_tile_id(scene_id: str) -> Optional[str]:
    """The MGRS tile (e.g. "43RGM") in a Sentinel-2 scene ID, or None when it cannot be parsed."""
    match = _MGRS_TILE.search(scene_id or "")
    return match.group(1) if match else None


def strips(height: int, rows: int = params.STRIP_ROWS) -> Iterator[Tuple[int, int]]:
    """Yield (row0, n_rows) covering [0, height)."""
    for row0 in range(0, height, rows):
        yield row0, min(rows, height - row0)


def dn_scale(dtype: str) -> float:
    """Digital-number range of a raster, used to express thresholds as a fraction of reflectance."""
    if dtype == "uint16":
        return 10000.0  # Sentinel-2 L2A quantification value
    if dtype == "uint8":
        return 255.0
    return 1.0


@dataclass
class GridSignature:
    crs: str
    transform: Tuple[float, ...]
    width: int
    height: int


def grid_signature(path: Path) -> GridSignature:
    with rasterio.open(str(path)) as src:
        return GridSignature(
            crs=src.crs.to_string() if src.crs else "",
            transform=tuple(src.transform)[:6],
            width=src.width,
            height=src.height,
        )


def grids_match(a: GridSignature, b: GridSignature) -> bool:
    """CRS, pixel origin, affine transform (hence resolution) and shape must all be identical.

    Matching CRS alone is not enough: two scenes in one UTM zone can have origins offset by a fraction of a
    pixel, and differencing them injects spurious change along every high-contrast edge.
    """
    if a.crs != b.crs or (a.width, a.height) != (b.width, b.height):
        return False
    return all(abs(x - y) <= 1e-6 * max(1.0, abs(x)) for x, y in zip(a.transform, b.transform))


def _matrix_rows_needed(matrix: np.ndarray, row0: int, rows: int, width: int) -> Tuple[float, float]:
    """Source-row range [lo, hi] that the output strip samples when resampled through `matrix` (A -> B)."""
    corners = np.array([[0, row0, 1], [width, row0, 1], [0, row0 + rows, 1], [width, row0 + rows, 1]], dtype=np.float64).T
    mapped = matrix @ corners
    ys = mapped[1] / mapped[2]
    return float(ys.min()), float(ys.max())


def warp_rows(
    read_rows: Callable[[int, int], np.ndarray],
    matrix: np.ndarray,
    row0: int,
    rows: int,
    width: int,
    in_height: int,
    interpolation: int = cv2.INTER_LINEAR,
    dtype: type = np.float32,
) -> np.ndarray:
    """Resample one output strip of scene B onto scene A's grid.

    `matrix` (3x3) maps A pixel coordinates to B pixel coordinates. `read_rows(lo, hi)` returns the (C, hi-lo, W)
    source rows. Only the source rows the strip actually samples are read.
    """
    lo_f, hi_f = _matrix_rows_needed(matrix, row0, rows, width)
    lo = max(0, int(np.floor(lo_f)) - 2)
    hi = min(in_height, int(np.ceil(hi_f)) + 3)
    channels = read_rows(lo, hi).shape[0] if hi > lo else 1
    out = np.zeros((channels, rows, width), dtype=dtype)
    if hi <= lo:
        return out

    src = read_rows(lo, hi)
    t_out = np.array([[1, 0, 0], [0, 1, row0], [0, 0, 1]], dtype=np.float64)  # local strip row -> full row
    t_in = np.array([[1, 0, 0], [0, 1, -lo], [0, 0, 1]], dtype=np.float64)  # full source row -> local source row
    local = t_in @ matrix @ t_out
    for c in range(src.shape[0]):
        out[c] = cv2.warpPerspective(
            np.ascontiguousarray(src[c]),
            local,
            (width, rows),
            flags=interpolation | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return out


class SceneReader:
    """Strip-wise reader for a scene's analysis raster, optionally resampled onto the reference grid.

    With `matrix=None` the raster is read as-is (it already shares the reference grid). Otherwise every strip is
    warped through the 3x3 A->B matrix found in Phase 2 with bilinear interpolation.
    """

    def __init__(self, path: Path, matrix: Optional[np.ndarray] = None) -> None:
        self.path = Path(path)
        self.matrix = matrix
        self._src = rasterio.open(str(self.path))
        self.width = self._src.width
        self.height = self._src.height
        self.count = self._src.count
        self.dtype = self._src.dtypes[0]
        self.tags = self._src.tags()

    def close(self) -> None:
        self._src.close()

    def __enter__(self) -> "SceneReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _read(self, lo: int, hi: int) -> np.ndarray:
        return self._src.read(window=Window(0, lo, self.width, hi - lo)).astype(np.float32)

    def strip(self, row0: int, rows: int) -> np.ndarray:
        """(bands, rows, width) float32 for the output strip on the reference grid."""
        if self.matrix is None:
            return self._read(row0, row0 + rows)
        return warp_rows(self._read, self.matrix, row0, rows, self.width, self.height)

    def data_mask(self, row0: int, rows: int) -> np.ndarray:
        """True where every band carries data (dataset nodata respected)."""
        masks = self._src.read_masks(window=Window(0, row0, self.width, rows))
        return (masks > 0).all(axis=0)


def channel(strip: np.ndarray) -> np.ndarray:
    """The band that is differenced: NIR (B08, band 4) when available, otherwise mean intensity."""
    if strip.shape[0] >= 4:
        return strip[3]
    return strip.mean(axis=0)


def uses_nir(count: int) -> bool:
    return count >= 4


def matrix_to_full(matrix: np.ndarray, factor: int) -> np.ndarray:
    """Convert an A->B matrix estimated on an image decimated by `factor` to full resolution.

    Pixel-centre convention: p_full = factor * (p_level + 0.5) - 0.5.
    """
    d = np.array([[factor, 0, 0.5 * factor - 0.5], [0, factor, 0.5 * factor - 0.5], [0, 0, 1]], dtype=np.float64)
    return d @ matrix @ np.linalg.inv(d)


def matrix_from_full(matrix_full: np.ndarray, factor: int) -> np.ndarray:
    """Inverse of matrix_to_full."""
    d = np.array([[factor, 0, 0.5 * factor - 0.5], [0, factor, 0.5 * factor - 0.5], [0, 0, 1]], dtype=np.float64)
    return np.linalg.inv(d) @ matrix_full @ d


def max_corner_shift(matrix: np.ndarray, width: int, height: int) -> float:
    """Largest displacement (pixels) the transform applies to the image corners and centre."""
    pts = np.array(
        [[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1], [width / 2, height / 2, 1]], dtype=np.float64
    ).T
    mapped = matrix @ pts
    mapped = mapped[:2] / mapped[2]
    return float(np.linalg.norm(mapped - pts[:2], axis=0).max())


def bands_summary(count: int) -> List[str]:
    return ["B02", "B03", "B04", "B08"] if count >= 4 else [f"band{i + 1}" for i in range(count)]
