"""IRIS change detection — Phase 3: radiometric normalisation (simplified PIF).

Pseudo-invariant pixels are the most stable ones: those whose mean absolute difference across bands is in the
bottom PIF_PERCENTILE of the mutual valid area. A per-band linear regression fitted on them maps scene B onto
scene A's radiometric baseline. Accumulating sums instead of collecting pixels keeps memory bounded.

Deliberate change to the simplified spec: selecting on |A - B| alone is a selection on the outcome. It keeps
pixels where A and B happen to be equal, which pulls the fitted gain towards 1 and hides a real radiometric
difference (measured on a real 10980 px scene: a true gain of 0.952 was fitted as 1.05, making B worse, not
better). Re-selecting around a line that was fitted on the biased set only reinforces it. So the fit starts from
a plain least-squares line over a sample of ALL mutual-valid pixels (independent of any selection), and the
pseudo-invariant pixels are then the bottom PIF_PERCENTILE by RESIDUAL to the current line. With the identity
line that is exactly the spec'd |A - B|.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from change_detection import params
from change_detection.rasters import SceneReader, strips

logger = logging.getLogger("iris.change.radiometry")


@dataclass
class Normalization:
    slope: np.ndarray  # per band
    intercept: np.ndarray
    n_pixels: int  # PIF pixels the final regression used
    threshold: float  # residual value that separates the final PIF set
    tier: str  # "pif", or "identity" when there were too few stable pixels to fit anything
    iterations: int = 0
    initial_slope: Optional[List[float]] = None  # the selection-free least-squares start, kept for the decision trace
    band_notes: List[str] = field(default_factory=list)

    def apply(self, strip: np.ndarray) -> np.ndarray:
        """Normalise a (bands, rows, width) strip of scene B: B' = slope * B + intercept."""
        return strip * self.slope[:, None, None].astype(np.float32) + self.intercept[:, None, None].astype(np.float32)

    def describe(self) -> Dict[str, object]:
        return {
            "tier": self.tier,
            "pif_pixels": self.n_pixels,
            "pif_threshold": round(float(self.threshold), 4),
            "iterations": self.iterations,
            "slope": [round(float(v), 5) for v in self.slope],
            "intercept": [round(float(v), 3) for v in self.intercept],
            "initial_slope": self.initial_slope,
            "notes": self.band_notes,
        }


def _identity(bands: int, reason: str, n: int = 0, thr: float = 0.0) -> Normalization:
    logger.warning("Radiometric normalisation skipped (%s); scene B is used as-is", reason)
    return Normalization(np.ones(bands), np.zeros(bands), n, thr, "identity", band_notes=[reason])


def _residual(a: np.ndarray, b: np.ndarray, slope: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    """Mean over bands of |A - (slope * B + intercept)|. With the identity fit this is the spec'd |A - B|."""
    fitted = b * slope[:, None, None].astype(np.float32) + intercept[:, None, None].astype(np.float32)
    return np.abs(a - fitted).mean(axis=0)


def _sample_strips(reader_a: SceneReader, mutual: np.ndarray) -> List[Tuple[int, int]]:
    """A few evenly spaced strips that contain valid pixels: enough to estimate a fit or a percentile."""
    all_strips = [(r0, n) for r0, n in strips(reader_a.height) if mutual[r0 : r0 + n].any()]
    step = max(1, len(all_strips) // params.PIF_SAMPLE_STRIPS)
    return all_strips[::step]


def _sample_threshold(
    reader_a: SceneReader,
    reader_b: SceneReader,
    mutual: np.ndarray,
    slope: np.ndarray,
    intercept: np.ndarray,
    sample: List[Tuple[int, int]],
) -> Optional[float]:
    """Percentile of the residual, estimated from the sampled strips rather than a full pass over the scene."""
    if not sample:
        return None
    samples = []
    for r0, n in sample:
        m = mutual[r0 : r0 + n]
        res = _residual(reader_a.strip(r0, n), reader_b.strip(r0, n), slope, intercept)
        samples.append(res[m])
    values = np.concatenate(samples)
    if values.size < params.PIF_MIN_PIXELS:
        return None
    return float(np.percentile(values, params.PIF_PERCENTILE))


def _fit_pass(
    reader_a: SceneReader,
    reader_b: SceneReader,
    mutual: np.ndarray,
    slope: np.ndarray,
    intercept: np.ndarray,
    threshold: float,
    only: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sufficient statistics (n, sum a, sum b, sum b^2, sum ab) of the pixels whose residual is <= threshold.

    `only` restricts the pass to the given (row0, rows) strips; None means the whole scene.
    """
    bands = reader_a.count
    n = 0
    sum_a, sum_b, sum_bb, sum_ab = (np.zeros(bands) for _ in range(4))
    for row0, rows in only if only is not None else strips(reader_a.height):
        m = mutual[row0 : row0 + rows]
        if not m.any():
            continue
        a = reader_a.strip(row0, rows)
        b = reader_b.strip(row0, rows)
        sel = m & (_residual(a, b, slope, intercept) <= threshold)
        k = int(sel.sum())
        if k == 0:
            continue
        av = a[:, sel].astype(np.float64)
        bv = b[:, sel].astype(np.float64)
        n += k
        sum_a += av.sum(axis=1)
        sum_b += bv.sum(axis=1)
        sum_bb += (bv * bv).sum(axis=1)
        sum_ab += (av * bv).sum(axis=1)
    return n, sum_a, sum_b, sum_bb, sum_ab


def _solve(n: int, sum_a, sum_b, sum_bb, sum_ab, bands: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Closed-form per-band regression A = slope * B + intercept, with plausibility gates."""
    slope = np.ones(bands)
    intercept = np.zeros(bands)
    notes: List[str] = []
    lo, hi = params.PIF_SLOPE_RANGE
    for i in range(bands):
        var_b = sum_bb[i] / n - (sum_b[i] / n) ** 2
        cov_ab = sum_ab[i] / n - (sum_a[i] / n) * (sum_b[i] / n)
        if var_b <= 1e-9:
            notes.append(f"band {i + 1}: PIF pixels have no variance; left unchanged")
            continue
        s = cov_ab / var_b
        if not (lo <= s <= hi):
            notes.append(f"band {i + 1}: fitted gain {s:.3f} outside [{lo}, {hi}]; left unchanged")
            continue
        slope[i] = s
        intercept[i] = sum_a[i] / n - s * sum_b[i] / n
    return slope, intercept, notes


def fit_normalization(reader_a: SceneReader, reader_b: SceneReader, mutual: np.ndarray) -> Normalization:
    """Fit B -> A per band on pseudo-invariant pixels within the mutual validity mask."""
    bands = reader_a.count
    if int(mutual.sum()) < params.PIF_MIN_PIXELS:
        return _identity(bands, "too few valid pixels")

    sample = _sample_strips(reader_a, mutual)

    # Start from a least-squares line over ALL mutual-valid pixels of the sampled strips: no selection, so no
    # selection bias. (Clouds are already out of the mask; genuine change is a small minority of pixels.)
    n0, sa, sb, sbb, sab = _fit_pass(
        reader_a, reader_b, mutual, np.ones(bands), np.zeros(bands), float("inf"), only=sample
    )
    if n0 < params.PIF_MIN_PIXELS:
        return _identity(bands, f"only {n0} valid pixels to fit", n0)
    slope, intercept, notes = _solve(n0, sa, sb, sbb, sab, bands)
    initial = [round(float(v), 5) for v in slope]

    # Then keep the bottom PIF_PERCENTILE by residual to that line and refit, until the gain stops moving
    n, threshold, done = n0, 0.0, 0
    for iteration in range(params.PIF_ITERATIONS):
        thr = _sample_threshold(reader_a, reader_b, mutual, slope, intercept, sample)
        if thr is None:
            return _identity(bands, "too few valid pixels to estimate the PIF threshold")
        n_sel, sa, sb, sbb, sab = _fit_pass(reader_a, reader_b, mutual, slope, intercept, thr)
        if n_sel < params.PIF_MIN_PIXELS:
            return _identity(bands, f"only {n_sel} pseudo-invariant pixels", n_sel, thr)
        new_slope, new_intercept, notes = _solve(n_sel, sa, sb, sbb, sab, bands)
        n, threshold, done = n_sel, thr, iteration + 1
        moved = not np.allclose(new_slope, slope, atol=1e-3)
        slope, intercept = new_slope, new_intercept
        if not moved:
            break

    logger.info(
        "PIF normalisation: %d pixels after %d iteration(s); slope %s (initial fit %s)", n, done, np.round(slope, 4), initial
    )
    return Normalization(slope, intercept, n, threshold, "pif", done, initial, notes)
