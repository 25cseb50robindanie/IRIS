"""IRIS change detection — merging nearby Change Blobs into coherent detections, and their geometry.

One construction site rarely arrives as one blob: roofs, roads and cleared ground come out of K-means as many separate
components, and listing each as a candidate buries the analyst. So, after connected-component labelling:

  1. assign_groups   dilate the mask of surviving blobs and label the dilated mask again; blobs that land in the same
                     dilated component share a group id. Tried widest first: a wide pass (LARGE_AREA_DILATIONS) that
                     merges a group only when it is a large-area change (3+ blobs, or over 500 px), then a near pass
                     (NEAR_DILATIONS) that merges any close blobs. At every level a group must still be one place
                     (bounding box and area within LARGE_AREA_MAX_*), or its blobs fall back to a narrower level:
                     merging is transitive, and on a dense scene it would otherwise link the whole tile. The dilated
                     masks are used for nothing else: blob pixels, areas and outlines come from the undilated mask.
  2. merge_candidates  once every blob is scored and has a direction, one candidate per group: combined bounding box,
                     total area, the dominant change type among its blobs, the highest confidence among them.
  3. attach_geometry  the outline of the changed pixels (the actual contour, not a box) and its centroid, in WGS84.

Dilating by N pixels merges blobs whose gap is up to 2N pixels, because both sides grow.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from rasterio import features
from rasterio.transform import Affine
from rasterio.warp import transform as warp_points, transform_bounds, transform_geom

from change_detection import params
from change_detection.rasters import strips
from mgrs_ref import to_mgrs
from scipy import ndimage

logger = logging.getLogger("iris.change.grouping")


def _group_map(labels: np.ndarray, keep: np.ndarray, radius: int) -> np.ndarray:
    """Group id per blob label: blobs whose masks touch after growing by `radius` pixels share an id (0 = no group)."""
    height, width = labels.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for row0, rows in strips(height):
        mask[row0 : row0 + rows] = keep[labels[row0 : row0 + rows]]  # keep[0] is False: background stays 0

    size = 2 * radius + 1
    dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    del mask
    grouped, _ = ndimage.label(dilated)  # same 4-connectivity as the blobs themselves
    del dilated

    # Each blob lies inside exactly one dilated component (dilation cannot split a connected set), so any pixel of
    # the blob names its group
    group_of = np.zeros(len(keep), dtype=np.int32)
    for row0, rows in strips(height):
        lab = labels[row0 : row0 + rows]
        sel = keep[lab]
        if sel.any():
            group_of[lab[sel]] = grouped[row0 : row0 + rows][sel]
    return group_of


def assign_groups(det: Any) -> Dict[str, Any]:
    """Give every surviving blob a group id (`det.group_of[label]`, 0 for anything that did not survive).

    The mask of surviving blobs is grown and re-labelled at several radii, widest first:
      large  LARGE_AREA_DILATIONS: a group merges only when it is a large-area change (at least LARGE_AREA_MIN_BLOBS
             blobs, or more than LARGE_AREA_MIN_PIXELS pixels).
      near   NEAR_DILATIONS: blobs that are close merge, whatever their size.
    At every level a group must also still be one place: its bounding box and area within LARGE_AREA_MAX_*. Chaining
    would otherwise link a whole dense scene into one "detection". A blob takes the widest level whose group is
    acceptable, and stays a candidate on its own if none is. Groups at a narrower radius sit inside those at a wider
    one, so the levels never disagree.
    """
    trace: Dict[str, Any] = {
        "dilation_px": params.GROUP_DILATION_PX,
        "near_dilations_px": list(params.NEAR_DILATIONS),
        "large_area_dilations_px": list(params.LARGE_AREA_DILATIONS),
        "blobs": 0,
        "groups": 0,
        "merged_groups": 0,
        "large_area_groups": 0,
    }
    if det.labels is None or det.keep is None or not det.keep.any():
        return trace

    labels, keep = det.labels, det.keep
    kept = np.flatnonzero(keep)
    group_of = np.arange(len(keep), dtype=np.int64)  # every blob starts as its own detection
    group_of[~keep] = 0
    radius_of = np.zeros(len(keep), dtype=np.int16)
    taken = np.zeros(len(keep), dtype=bool)
    next_id = len(keep)

    # bounding box of every blob, for the size guard
    rows0 = np.array([det.objects[k - 1][0].start for k in kept])
    rows1 = np.array([det.objects[k - 1][0].stop for k in kept])
    cols0 = np.array([det.objects[k - 1][1].start for k in kept])
    cols1 = np.array([det.objects[k - 1][1].stop for k in kept])
    area = det.areas[kept].astype(np.float64)

    rejected = 0
    levels = [(r, True) for r in params.LARGE_AREA_DILATIONS] + [(r, False) for r in params.NEAR_DILATIONS]
    for radius, needs_size in levels:
        if taken[kept].all():
            break
        grouped = _group_map(labels, keep, radius)
        ids = grouped[kept]
        n = int(ids.max()) + 1
        count = np.bincount(ids, minlength=n)
        pixels = np.bincount(ids, weights=area, minlength=n)
        top, left = np.full(n, np.inf), np.full(n, np.inf)
        bottom, right = np.full(n, -np.inf), np.full(n, -np.inf)
        np.minimum.at(top, ids, rows0)
        np.minimum.at(left, ids, cols0)
        np.maximum.at(bottom, ids, rows1)
        np.maximum.at(right, ids, cols1)
        extent = np.maximum(bottom - top, right - left)

        wanted = count >= 2  # only a group of at least two blobs is a merge at all
        if needs_size:
            wanted &= (count >= params.LARGE_AREA_MIN_BLOBS) | (pixels > params.LARGE_AREA_MIN_PIXELS)
        one_place = (extent <= params.LARGE_AREA_MAX_EXTENT_PX) & (pixels <= params.LARGE_AREA_MAX_PIXELS)
        rejected += int((wanted & ~one_place).sum())

        take = (wanted & one_place)[ids] & ~taken[kept]
        group_of[kept[take]] = ids[take] + next_id
        radius_of[kept[take]] = radius
        taken[kept[take]] = True
        next_id += n
        del grouped

    det.group_of = group_of.astype(np.int64)
    det.group_radius = radius_of
    det.group_large = radius_of > params.GROUP_DILATION_PX
    per_group = np.bincount(np.unique(group_of[kept], return_inverse=True)[1])
    large_groups = len(np.unique(group_of[kept[det.group_large[kept]]]))
    trace.update(
        {
            "blobs": int(keep.sum()),
            "groups": int(per_group.size),
            "merged_groups": int((per_group > 1).sum()),
            "large_area_groups": large_groups,
            "too_large_to_merge": rejected,  # groups that failed the size guard and fell back to a narrower level
            "largest_group_blobs": int(per_group.max()),
        }
    )
    logger.info(
        "Grouping: %d blobs -> %d detections (%d merged, %d large-area, %d groups over the size guard)",
        trace["blobs"],
        trace["groups"],
        trace["merged_groups"],
        trace["large_area_groups"],
        rejected,
    )
    return trace


def _dominant(members: List[Dict[str, Any]]) -> Tuple[str, str, Dict[str, Any]]:
    """(direction, change_type, evidence) that covers the most area among a group's blobs.

    "unclassified" means no rule matched, not a competing class, so a named type outranks it whatever its area; a
    group is unclassified only when none of its blobs is named. Ties go to the blob with the higher confidence.
    """

    def area_by(key: str, pool: List[Dict[str, Any]]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for m in pool:
            out[m[key]] = out.get(m[key], 0) + int(m["area_px"])
        return out

    def pick(totals: Dict[str, int], key: str, pool: List[Dict[str, Any]]) -> str:
        return max(totals, key=lambda t: (totals[t], max(m["confidence"] for m in pool if m[key] == t)))

    typed = [m for m in members if m["change_type"] != "unclassified"]
    if typed:
        change_type = pick(area_by("change_type", typed), "change_type", typed)
        pool = [m for m in typed if m["change_type"] == change_type]
        direction = pick(area_by("direction", pool), "direction", pool)
        pool = [m for m in pool if m["direction"] == direction]
    else:
        directed = [m for m in members if m["direction"] not in (None, "unclassified")]
        change_type = "unclassified"
        if directed:
            direction = pick(area_by("direction", directed), "direction", directed)
            pool = [m for m in directed if m["direction"] == direction]
        else:
            direction, pool = "unclassified", members
    lead = max(pool, key=lambda m: (m["area_px"], m["confidence"]))
    return direction, change_type, dict(lead["direction_evidence"] or {})


def merge_candidates(rows: List[Dict[str, Any]], crs: Optional[str]) -> List[Dict[str, Any]]:
    """One candidate per group of scored, direction-classified blob rows."""
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(r.get("group_id") or -(i + 1), []).append(r)  # no group id: stands alone

    merged: List[Dict[str, Any]] = []
    for members in groups.values():
        best = max(members, key=lambda m: (m["confidence"], m["area_px"]))
        out = dict(best)  # confidence and its four terms travel together, so the breakdown still sums to the score
        out["blob_labels"] = [int(m["blob_label"]) for m in members]
        out["sub_blobs"] = len(members)
        if len(members) > 1:
            out["min_x"] = min(m["min_x"] for m in members)
            out["min_y"] = min(m["min_y"] for m in members)
            out["max_x"] = max(m["max_x"] for m in members)
            out["max_y"] = max(m["max_y"] for m in members)
            wgs = [min(m["min_lon"] for m in members), min(m["min_lat"] for m in members),
                   max(m["max_lon"] for m in members), max(m["max_lat"] for m in members)]
            if crs:
                try:
                    wgs = [float(v) for v in transform_bounds(crs, "EPSG:4326", out["min_x"], out["min_y"], out["max_x"], out["max_y"])]
                except Exception:
                    logger.warning("Could not reproject merged bounds to WGS84; using the union of blob bounds")
            out["min_lon"], out["min_lat"], out["max_lon"], out["max_lat"] = wgs
            out["area_px"] = int(sum(m["area_px"] for m in members))
            with_ndvi = [m for m in members if m["mean_dndvi"] is not None]
            weight = sum(m["area_px"] for m in with_ndvi)
            out["mean_dndvi"] = sum(m["mean_dndvi"] * m["area_px"] for m in with_ndvi) / weight if weight else None
            direction, change_type, evidence = _dominant(members)
            out["direction"], out["change_type"] = direction, change_type
            if evidence:
                evidence["merged"] = {
                    "sub_blobs": len(members),
                    "large_area": any(m.get("group_large") for m in members),
                    "dilation_px": max(int(m.get("group_radius") or 0) for m in members),
                    "dominant_by": "area",
                    "area_by_type": {
                        t: int(sum(m["area_px"] for m in members if m["change_type"] == t))
                        for t in sorted({m["change_type"] for m in members})
                    },
                }
            out["direction_evidence"] = evidence or None
        out.pop("blob_label", None)
        out.pop("group_id", None)
        out.pop("group_large", None)
        out.pop("group_radius", None)
        merged.append(out)
    merged.sort(key=lambda r: (-r["confidence"], -r["area_px"]))
    return merged


def _orient(ring: List[List[float]], counter_clockwise: bool) -> List[List[float]]:
    """RFC 7946 winding: exterior rings counter-clockwise, holes clockwise."""
    area2 = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:]))
    return ring if (area2 > 0) == counter_clockwise else ring[::-1]


def attach_geometry(det: Any, rows: List[Dict[str, Any]], transform: Affine, crs: Optional[str]) -> None:
    """Fill `geometry` (GeoJSON, EPSG:4326), `centroid_lon/lat` and `mgrs_ref` on each row, in place.

    The outline is traced on the original label raster, so it follows the changed pixels rather than the box around them.
    """
    for row in rows:
        labels = row.pop("blob_labels", None)
        if labels is None or det.labels is None:
            continue
        slices = [det.objects[label - 1] for label in labels]
        r0, r1 = min(s[0].start for s in slices), max(s[0].stop for s in slices)
        c0, c1 = min(s[1].start for s in slices), max(s[1].stop for s in slices)
        mask = np.isin(np.asarray(det.labels[r0:r1, c0:c1]), labels)
        if not mask.any():
            continue

        window_transform = transform * Affine.translation(c0, r0)
        polygons = [
            geom["coordinates"]
            for geom, _ in features.shapes(mask.astype(np.uint8), mask=mask, transform=window_transform, connectivity=4)
        ]
        geometry: Dict[str, Any] = {"type": "MultiPolygon", "coordinates": polygons}
        if crs:
            geometry = transform_geom(crs, "EPSG:4326", geometry, precision=6)
        for poly in geometry["coordinates"]:
            for i, ring in enumerate(poly):
                poly[i] = _orient([list(p) for p in ring], counter_clockwise=(i == 0))
        if len(geometry["coordinates"]) == 1:
            geometry = {"type": "Polygon", "coordinates": geometry["coordinates"][0]}
        row["geometry"] = json.dumps(geometry, separators=(",", ":"))

        ys, xs = np.nonzero(mask)
        x, y = window_transform * (float(xs.mean()) + 0.5, float(ys.mean()) + 0.5)
        lon, lat = x, y
        if crs:
            (lons, lats) = warp_points(crs, "EPSG:4326", [x], [y])
            lon, lat = float(lons[0]), float(lats[0])
        lon, lat = round(lon, 6), round(lat, 6)
        row["centroid_lon"], row["centroid_lat"] = lon, lat
        row["mgrs_ref"] = to_mgrs(lat, lon)  # from the stored values, so the two always agree
