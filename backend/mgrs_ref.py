"""MGRS grid references for WGS84 points (pure-Python `mgrs` package, no network).

Precision 5 is the 10-digit reference (1 m). It is a label for where a point is, not a claim that the point is
known to 1 m: a tile centroid is the middle of a ~5 km box. The frontend uses the same precision (frontend/src/mgrs.js).
"""

import logging
from typing import List, Optional, Sequence

import mgrs

logger = logging.getLogger("iris.mgrs")

MGRS_PRECISION = 5  # digits per axis: 5 = 10-digit grid reference

_converter = mgrs.MGRS()


def to_mgrs(lat: float, lon: float, precision: int = MGRS_PRECISION) -> Optional[str]:
    """MGRS reference for a WGS84 point, or None when it cannot be expressed (outside 80S-84N, bad values)."""
    try:
        if not (-80.0 <= float(lat) <= 84.0 and -180.0 <= float(lon) <= 180.0):
            return None
        return _converter.toMGRS(float(lat), float(lon), MGRSPrecision=precision)
    except Exception:
        logger.debug("No MGRS reference for lat=%s lon=%s", lat, lon)
        return None


def bounds_centre_mgrs(bounds: Optional[Sequence[float]]) -> Optional[str]:
    """MGRS reference of the centre of a WGS84 [min_lon, min_lat, max_lon, max_lat] box."""
    if not bounds or len(bounds) != 4 or any(b is None for b in bounds):
        return None
    return to_mgrs((bounds[1] + bounds[3]) / 2.0, (bounds[0] + bounds[2]) / 2.0)


def backfill_missing(rows: List[tuple]) -> List[tuple]:
    """[(key, min_lon, min_lat, max_lon, max_lat)] -> [(mgrs, key)] for the rows a reference can be made for."""
    out = []
    for key, min_lon, min_lat, max_lon, max_lat in rows:
        ref = bounds_centre_mgrs([min_lon, min_lat, max_lon, max_lat])
        if ref:
            out.append((ref, key))
    return out
