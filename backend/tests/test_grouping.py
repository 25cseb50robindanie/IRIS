"""Tests for merging nearby Change Blobs into detections, their outlines, MGRS references and the display window."""

import json
import math
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api.provenance import display_bounds, intersect
from catalog import changes
from catalog.database import init_connection, init_schema
from change_detection import params, run_change_detection
from change_detection.grouping import _dominant, assign_groups, merge_candidates
from mgrs_ref import bounds_centre_mgrs, to_mgrs
from s2_factory import Scenario, ingest_without_embedding, make_safe

NAME_A = "S2A_MSIL2A_20240110T050649_N0510_R019_T43PHM_20240110T090000"
NAME_B = "S2A_MSIL2A_20240214T050649_N0510_R019_T43PHM_20240214T090000"


# ---- assign_groups -----------------------------------------------------------------------------------------------


def _detection(mask: np.ndarray, min_pixels: int = 1):
    labels, n = ndimage.label(mask)
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    keep = areas >= min_pixels
    keep[0] = False
    return SimpleNamespace(
        labels=labels.astype(np.int32), keep=keep, areas=areas, objects=ndimage.find_objects(labels),
        group_of=None, group_large=None, group_radius=None,
    )


@pytest.fixture
def near_only(monkeypatch):
    """Switch the large-area pass off, to test the near pass on its own."""
    monkeypatch.setattr(params, "LARGE_AREA_MIN_BLOBS", 10**6)
    monkeypatch.setattr(params, "LARGE_AREA_MIN_PIXELS", 10**12)


def test_blobs_within_the_grown_distance_share_a_group_and_far_ones_do_not():
    mask = np.zeros((200, 400), dtype=np.uint8)
    mask[50:80, 20:60] = 1  # blob 1
    mask[50:80, 70:110] = 1  # blob 2: 10 px from blob 1, well inside 2 x 10
    mask[50:80, 200:240] = 1  # blob 3: 90 px away
    det = _detection(mask)
    trace = assign_groups(det)

    g = det.group_of
    assert g[1] == g[2] != g[3] > 0
    assert trace["blobs"] == 3 and trace["groups"] == 2 and trace["merged_groups"] == 1
    assert trace["dilation_px"] == params.GROUP_DILATION_PX == 10


def test_the_merge_distance_is_twice_the_dilation(near_only):
    # gaps of 18 px merge (both blobs grow 10), gaps of 24 px do not
    for gap, merged in ((18, True), (24, False)):
        mask = np.zeros((120, 300), dtype=np.uint8)
        mask[40:70, 20:60] = 1
        mask[40:70, 60 + gap : 100 + gap] = 1
        det = _detection(mask)
        assign_groups(det)
        assert bool(det.group_of[1] == det.group_of[2]) is merged, gap


# ---- the large-area pass (30 px dilation; 3+ blobs or over 500 px) -----------------------------------------------


def blobs(*specs, shape=(200, 500)):
    """A mask with one rectangle per (row0, row1, col0, col1)."""
    mask = np.zeros(shape, dtype=np.uint8)
    for r0, r1, c0, c1 in specs:
        mask[r0:r1, c0:c1] = 1
    return _detection(mask)


def test_three_small_blobs_within_range_form_one_large_area_group():
    det = blobs((50, 60, 20, 30), (50, 60, 60, 70), (50, 60, 100, 110))  # 100 px each, 30 px gaps: 300 px in all
    trace = assign_groups(det)
    assert det.group_of[1] == det.group_of[2] == det.group_of[3] > 0  # 3 blobs qualifies although 300 < 500 px
    assert det.group_large[1:4].all() and trace["groups"] == 1 and trace["large_area_groups"] == 1
    assert trace["large_area_dilations_px"] == list(params.LARGE_AREA_DILATIONS) == [30, 20]


def test_two_small_blobs_in_the_wide_range_stay_separate():
    det = blobs((50, 60, 20, 30), (50, 60, 60, 70))  # two blobs, 200 px in all: not a large-area change
    assign_groups(det)
    assert det.group_of[1] != det.group_of[2] and not det.group_large.any()


def test_two_blobs_over_500_px_merge_but_exactly_500_does_not():
    over = blobs((50, 76, 20, 30), (50, 75, 60, 70))  # 260 + 250 = 510 px, 30 px gap
    assign_groups(over)
    assert over.group_of[1] == over.group_of[2] and over.group_large[1]
    exactly = blobs((50, 75, 20, 30), (50, 75, 60, 70))  # 250 + 250 = 500 px: "more than 500" is strict
    assign_groups(exactly)
    assert exactly.group_of[1] != exactly.group_of[2] and not exactly.group_large.any()


def test_the_wide_pass_reaches_60_px_gaps_and_no_further():
    for gap, merged in ((56, True), (66, False)):
        det = blobs((50, 90, 20, 60), (50, 90, 60 + gap, 100 + gap))  # 1600 px each: area is not the limit
        assign_groups(det)
        assert bool(det.group_of[1] == det.group_of[2]) is merged, gap


def test_a_chain_that_runs_too_far_is_not_one_change(monkeypatch):
    """Groups chain transitively: a dense scene would join everything. The size guard stops that."""
    monkeypatch.setattr(params, "LARGE_AREA_MAX_EXTENT_PX", 200)
    specs = [(50, 70, 20 + i * 60, 40 + i * 60) for i in range(8)]  # 8 blobs, 40 px gaps, a chain 460 px long
    det = blobs(*specs, shape=(120, 520))
    trace = assign_groups(det)
    # at 30 px the whole chain is one 460 px group (too long); at 20 px the gaps (40) still link it, so nothing large
    # qualifies, and every blob falls back to the near pass, where 40 px gaps do not merge
    assert trace["too_large_to_merge"] >= 1 and trace["large_area_groups"] == 0
    assert len(set(det.group_of[1:9])) == 8 and not det.group_large.any()


def test_a_group_that_is_too_wide_at_30_px_can_still_merge_at_20_px(monkeypatch):
    monkeypatch.setattr(params, "LARGE_AREA_MAX_EXTENT_PX", 150)
    # cluster A: 3 blobs 36 px apart (102 px long). Cluster B: 3 more, 36 px apart, 50 px beyond A (in reach at 30 px,
    # not at 20 px). Together they are 254 px long: too wide as one group, fine as two.
    a = [(50, 60, 20 + i * 46, 30 + i * 46) for i in range(3)]
    b = [(50, 60, 172 + i * 46, 182 + i * 46) for i in range(3)]
    det = blobs(*(a + b), shape=(120, 520))
    trace = assign_groups(det)
    g = det.group_of
    assert g[1] == g[2] == g[3] and g[4] == g[5] == g[6] and g[1] != g[4]  # two 3-blob changes, not one of six
    assert trace["large_area_groups"] == 2 and trace["too_large_to_merge"] >= 1
    assert (det.group_radius[1:7] == 20).all()


def test_the_area_limit_also_applies(monkeypatch):
    monkeypatch.setattr(params, "LARGE_AREA_MAX_PIXELS", 2000)
    det = blobs((50, 90, 20, 60), (50, 90, 100, 140), (50, 90, 180, 220))  # 3 x 1600 px = 4800 px, 40 px gaps
    trace = assign_groups(det)
    assert len(set(det.group_of[1:4])) == 3 and trace["too_large_to_merge"] >= 1


def test_the_near_pass_is_guarded_too_and_falls_back_to_a_narrower_dilation(monkeypatch):
    """The 10 px pass chained across a whole real scene (94,522 ha) before it had a size guard."""
    monkeypatch.setattr(params, "LARGE_AREA_MIN_BLOBS", 10**6)  # wide passes off: only the near levels are in play
    monkeypatch.setattr(params, "LARGE_AREA_MIN_PIXELS", 10**12)
    monkeypatch.setattr(params, "LARGE_AREA_MAX_EXTENT_PX", 100)
    tight = [(50, 60, 10 + i * 22, 20 + i * 22) for i in range(8)]  # 12 px gaps: 10 px links them; chain is 170 px long
    det = blobs(*tight, shape=(120, 300))
    trace = assign_groups(det)
    # 10 px links the whole 170 px chain (too long); 5 px does not link 12 px gaps, so every blob stays on its own
    assert len(set(det.group_of[1:9])) == 8 and trace["too_large_to_merge"] >= 1

    # a short chain of the same blobs is fine
    det = blobs(*tight[:4], shape=(120, 300))
    assign_groups(det)
    assert len(set(det.group_of[1:5])) == 1


def test_a_lone_large_blob_is_not_a_merge():
    det = blobs((20, 120, 20, 120))  # 10,000 px on its own
    trace = assign_groups(det)
    assert trace["groups"] == 1 and trace["merged_groups"] == 0 and trace["large_area_groups"] == 0
    assert not det.group_large.any()


def test_close_small_blobs_still_merge_in_the_near_pass_without_being_large_area():
    det = blobs((50, 60, 20, 30), (50, 60, 38, 48))  # 100 px each, 8 px apart, 200 px in all
    assign_groups(det)
    assert det.group_of[1] == det.group_of[2] and not det.group_large.any()


def test_a_qualifying_group_swallows_the_blobs_the_near_pass_had_joined():
    # two close blobs (near pass) plus a third 50 px away: 3 blobs in range, so all three become one detection
    det = blobs((50, 60, 20, 30), (50, 60, 38, 48), (50, 60, 100, 110))
    assign_groups(det)
    assert det.group_of[1] == det.group_of[2] == det.group_of[3] and det.group_large[1:4].all()


def test_grouping_never_changes_the_blob_pixels():
    mask = np.zeros((120, 200), dtype=np.uint8)
    mask[40:70, 20:60] = 1
    mask[40:70, 70:110] = 1
    det = _detection(mask)
    before = det.labels.copy()
    assign_groups(det)
    assert np.array_equal(det.labels, before)  # geometry comes from the undilated mask


def test_blobs_below_the_minimum_mapping_unit_do_not_join_or_bridge(near_only):
    mask = np.zeros((120, 300), dtype=np.uint8)
    mask[40:70, 20:60] = 1  # 1200 px
    mask[50:52, 68:70] = 1  # 4 px speck between the two big blobs
    mask[40:70, 80:120] = 1  # 1200 px, 20 px from the first
    det = _detection(mask, min_pixels=100)
    assign_groups(det)
    kept = np.flatnonzero(det.keep)
    assert len(kept) == 2 and det.group_of[kept[0]] == det.group_of[kept[1]]  # 20 px gap: merged on their own
    assert det.group_of[np.flatnonzero(~det.keep)].tolist() == [0, 0]  # background and the speck have no group


def test_no_blobs_means_no_groups():
    det = _detection(np.zeros((50, 50), dtype=np.uint8))
    trace = assign_groups(det)
    assert trace["groups"] == 0 and det.group_of is None


# ---- merge_candidates --------------------------------------------------------------------------------------------


def row(label, group, area, confidence, change_type="clearance", direction="disappearance", dndvi=-0.3, box=(0, 0, 10, 10), **kw):
    min_x, min_y, max_x, max_y = box
    base = {
        "blob_label": label,
        "group_id": group,
        "area_px": area,
        "confidence": confidence,
        "change_type": change_type,
        "direction": direction,
        "mean_dndvi": dndvi,
        "min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y,
        "min_lon": min_x / 1e5, "min_lat": min_y / 1e5, "max_lon": max_x / 1e5, "max_lat": max_y / 1e5,
        "direction_evidence": {"dominant_a": "vegetation", "dominant_b": "bare", "rule": "vegetation became bare"},
        "norm_cluster_dist": confidence, "terrain_flatness": 1.0, "valid_coverage": 1.0, "alignment_quality": 0.9,
    }
    base.update(kw)
    return base


def test_a_merged_detection_takes_the_union_box_total_area_and_best_confidence():
    rows = [
        row(1, 7, 300, 0.60, box=(0, 0, 10, 10)),
        row(2, 7, 500, 0.85, box=(20, 5, 40, 30)),
        row(3, 9, 200, 0.70, box=(500, 500, 510, 510)),
    ]
    out = merge_candidates(rows, crs=None)
    assert len(out) == 2
    big = next(r for r in out if r["sub_blobs"] == 2)
    assert (big["min_x"], big["min_y"], big["max_x"], big["max_y"]) == (0, 0, 40, 30)
    assert big["area_px"] == 800
    assert big["confidence"] == 0.85 and big["norm_cluster_dist"] == 0.85  # the best blob's score AND its terms
    assert sorted(big["blob_labels"]) == [1, 2] and "blob_label" not in big and "group_id" not in big
    assert big["mean_dndvi"] == pytest.approx(-0.3)
    single = next(r for r in out if r["sub_blobs"] == 1)
    assert single["blob_labels"] == [3] and single["area_px"] == 200
    assert [r["confidence"] for r in out] == sorted((r["confidence"] for r in out), reverse=True)


def test_dominant_type_is_by_area_and_a_named_type_outranks_unclassified():
    members = [
        row(1, 1, 100, 0.9, change_type="clearance", direction="disappearance"),
        row(2, 1, 400, 0.5, change_type="construction", direction="appearance", direction_evidence={"dominant_a": "bare", "dominant_b": "bare", "rule": "x"}),
        row(3, 1, 5000, 0.7, change_type="unclassified", direction="unclassified"),
    ]
    direction, change_type, evidence = _dominant(members)
    assert (direction, change_type) == ("appearance", "construction")  # 400 px beats 100 px; the 5000 px unclassified blob is not a rival
    assert evidence["dominant_a"] == "bare"

    only_unclassified = [row(1, 1, 50, 0.5, change_type="unclassified", direction="unclassified")]
    assert _dominant(only_unclassified)[:2] == ("unclassified", "unclassified")

    # named direction but no named type: the direction still counts
    directed = [
        row(1, 1, 100, 0.5, change_type="unclassified", direction="contraction"),
        row(2, 1, 300, 0.5, change_type="unclassified", direction="expansion"),
    ]
    assert _dominant(directed)[:2] == ("expansion", "unclassified")


def test_merged_evidence_records_how_the_type_was_chosen():
    rows = [row(1, 1, 300, 0.6), row(2, 1, 100, 0.8, change_type="construction", direction="appearance")]
    (merged,) = merge_candidates(rows, crs=None)
    assert merged["change_type"] == "clearance"  # 300 px vs 100 px
    info = merged["direction_evidence"]["merged"]
    assert info["sub_blobs"] == 2 and info["area_by_type"] == {"clearance": 300, "construction": 100}
    # the confidence is still the best blob's (0.8) even though the type is not
    assert merged["confidence"] == 0.8


def test_tied_area_goes_to_the_more_confident_type():
    rows = [row(1, 1, 200, 0.5, change_type="clearance"), row(2, 1, 200, 0.9, change_type="water_loss")]
    assert merge_candidates(rows, crs=None)[0]["change_type"] == "water_loss"


# ---- display window ----------------------------------------------------------------------------------------------


def test_display_window_is_four_times_the_box_and_centred():
    box = [73.0, 28.0, 73.01, 28.02]
    w = display_bounds(box, None, 4.0)
    assert (w[2] - w[0]) == pytest.approx(0.04) and (w[3] - w[1]) == pytest.approx(0.08)
    assert (w[0] + w[2]) / 2 == pytest.approx(73.005) and (w[1] + w[3]) / 2 == pytest.approx(28.01)
    inside = [72.0, 27.0, 74.0, 29.0]
    assert display_bounds(box, inside, 4.0) == pytest.approx(w)  # nothing to clamp: identical


def test_display_window_slides_in_from_the_edge_and_keeps_its_size():
    extent = [73.0, 28.0, 73.1, 28.1]
    box = [73.0, 28.04, 73.01, 28.06]  # touching the west edge
    w = display_bounds(box, extent, 4.0)
    assert w[0] == pytest.approx(73.0)  # pinned to the scene edge
    assert (w[2] - w[0]) == pytest.approx(0.04)  # still 4 boxes wide
    assert w[1] >= extent[1] and w[3] <= extent[3]
    assert w[0] <= box[0] + 1e-9 and w[2] >= box[2] - 1e-9  # the box is inside its window


def test_display_window_shrinks_to_a_small_extent():
    extent = [73.0, 28.0, 73.02, 28.02]
    w = display_bounds([73.005, 28.005, 73.015, 28.015], extent, 4.0)  # 4x = 0.04 > the 0.02 extent
    assert w == pytest.approx(extent)


def test_intersect():
    assert intersect([0, 0, 2, 2], [1, 1, 3, 3]) == [1, 1, 2, 2]
    assert intersect([0, 0, 1, 1], [2, 2, 3, 3]) is None
    assert intersect(None, [0, 0, 1, 1]) is None


# ---- MGRS --------------------------------------------------------------------------------------------------------


def test_mgrs_reference_is_ten_digits_and_matches_the_javascript_library():
    ref = to_mgrs(28.0, 73.0)
    assert ref == "43RCL0333298814"  # frontend/src/mgrs.js gives the same string for the same point
    assert len(ref) == 15 and ref[:5] == "43RCL" and ref[5:].isdigit()
    assert bounds_centre_mgrs([72.99, 27.99, 73.01, 28.01]) == ref
    assert to_mgrs(89.0, 0.0) is None and to_mgrs(28.0, 200.0) is None  # outside what MGRS can express
    assert bounds_centre_mgrs(None) is None


def test_tiles_get_an_mgrs_reference_and_old_databases_are_backfilled(tmp_path):
    db = tmp_path / "old.db"
    conn = init_connection(db)
    init_schema(conn)
    conn.execute("INSERT INTO scenes (scene_id, sensor, acquisition_date, crs, file_path) VALUES ('s', 'x', '2024-01-01', 'EPSG:4326', 'p')")
    conn.execute(
        "INSERT INTO tiles (tile_id, scene_id, min_x, min_y, max_x, max_y, min_lon, min_lat, max_lon, max_lat) "
        "VALUES ('t', 's', 0, 0, 1, 1, 72.99, 27.99, 73.01, 28.01)"
    )
    conn.commit()
    conn.execute("ALTER TABLE tiles DROP COLUMN mgrs_ref")  # a catalog from before MGRS geocoding
    conn.commit()
    init_schema(conn)
    assert conn.execute("SELECT mgrs_ref FROM tiles WHERE tile_id = 't'").fetchone()[0] == "43RCL0333298814"
    conn.close()


# ---- end to end: merging, outlines and MGRS through the real pipeline ---------------------------------------------

# three cleared patches 10 px apart in a row (one site), and a fourth far away
SITE = [(100, 160, 100, 180), (100, 160, 190, 270), (100, 160, 280, 360)]
FAR = (300, 360, 100, 180)


def run_site(tmp_path, boxes):
    a_cover = [(*b, "vegetation") for b in boxes]
    b_cover = [(*b, "bare") for b in boxes]
    a = ingest_without_embedding(make_safe(tmp_path / "src", NAME_A, Scenario(date="2024-01-10", noise_seed=1, landcover=a_cover)))
    b = ingest_without_embedding(make_safe(tmp_path / "src", NAME_B, Scenario(date="2024-02-14", noise_seed=2, landcover=b_cover)))
    run_change_detection(a, b)
    conn = init_connection()
    try:
        init_schema(conn)
        return (
            changes.list_candidates(conn),
            changes.get_job_for_pair(conn, a, b),
            dict(conn.execute("SELECT candidate_id, geometry FROM change_candidates").fetchall()),
        )
    finally:
        conn.close()


def ring_area_m2(ring):
    """Shoelace area of a lon/lat ring, in square metres (local scale; fine at this size)."""
    lat0 = math.radians(sum(p[1] for p in ring) / len(ring))
    pts = [(p[0] * 111320.0 * math.cos(lat0), p[1] * 110574.0) for p in ring]
    return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:])) / 2.0


def test_a_site_of_nearby_blobs_is_one_detection_and_a_far_one_stays_separate(tmp_path):
    cands, job, geometry = run_site(tmp_path, SITE + [FAR])

    assert len(cands) == 2, [(c["sub_blobs"], c["area_px"]) for c in cands]
    site = next(c for c in cands if c["sub_blobs"] == 3)
    alone = next(c for c in cands if c["sub_blobs"] == 1)

    # the site's box is the union of the three patches (260 x 60 px at 10 m), its area the sum of their pixels
    assert site["max_x"] - site["min_x"] == pytest.approx(2600, abs=40)
    assert site["max_y"] - site["min_y"] == pytest.approx(600, abs=40)
    assert site["area_px"] == pytest.approx(3 * 4800, rel=0.05)
    assert alone["area_px"] == pytest.approx(4800, rel=0.05)
    assert (site["direction"], site["change_type"]) == ("disappearance", "clearance")
    assert 0 < site["confidence"] <= 1

    trace = job["details"]["grouping"]
    assert trace["blobs"] == 4 and trace["groups"] == 2 and trace["merged_groups"] == 1 and trace["detections"] == 2
    assert job["details"]["scoring"]["candidates"] == 2 == job["details"]["scoring"]["candidates_found"]
    assert {"grouping", "merging"} <= set(job["details"]["timings_s"])

    # the outline is the changed pixels (three separate patches), not the dilated mask that joined them
    geom = json.loads(geometry[site["candidate_id"]])
    assert geom["type"] == "MultiPolygon" and len(geom["coordinates"]) == 3
    total = sum(ring_area_m2(poly[0]) for poly in geom["coordinates"])
    assert total == pytest.approx(site["area_px"] * 100.0, rel=0.05)  # 100 m^2 per 10 m pixel: no dilated fringe
    for poly in geom["coordinates"]:
        assert ring_area_m2(poly[0]) > 0  # exterior rings counter-clockwise (RFC 7946)
        lons, lats = zip(*poly[0])
        assert site["min_lon"] - 1e-6 <= min(lons) and max(lons) <= site["max_lon"] + 1e-6
        assert site["min_lat"] - 1e-6 <= min(lats) and max(lats) <= site["max_lat"] + 1e-6
    single = json.loads(geometry[alone["candidate_id"]])
    assert single["type"] == "Polygon"
    assert ring_area_m2(single["coordinates"][0]) == pytest.approx(alone["area_px"] * 100.0, rel=0.05)


def test_detections_carry_the_mgrs_of_their_centroid(tmp_path):
    cands, _, _ = run_site(tmp_path, [FAR])
    (c,) = cands
    assert c["centroid_lon"] is not None and c["min_lon"] < c["centroid_lon"] < c["max_lon"]
    assert c["min_lat"] < c["centroid_lat"] < c["max_lat"]
    assert c["mgrs_ref"] == to_mgrs(c["centroid_lat"], c["centroid_lon"])
    assert c["mgrs_ref"].startswith("43") and len(c["mgrs_ref"]) == 15


def test_a_blob_group_takes_its_best_blob_but_the_scale_still_spreads(tmp_path):
    """Confidence ranks one value per detection, so merging does not squeeze the scale toward the top."""
    a_cover = [(*b, "vegetation") for b in SITE + [FAR]]
    scene_a = Scenario(date="2024-01-10", noise_seed=1, landcover=a_cover)
    scene_b = Scenario(
        date="2024-02-14", noise_seed=2, landcover=[(*b, "bare") for b in SITE] + [(*FAR, "bare")]
    )
    a = ingest_without_embedding(make_safe(tmp_path / "src", NAME_A, scene_a))
    b = ingest_without_embedding(make_safe(tmp_path / "src", NAME_B, scene_b))
    run_change_detection(a, b)
    conn = init_connection()
    try:
        init_schema(conn)
        cands = changes.list_candidates(conn)
    finally:
        conn.close()
    pcts = sorted(c["norm_cluster_dist"] for c in cands)
    assert len(pcts) == 2 and pcts[0] < 0.5 < pcts[1]  # two detections rank 0.25 and 0.75, not both near the top


def test_the_old_schema_without_geometry_columns_is_migrated(tmp_path):
    db = tmp_path / "legacy.db"
    raw = sqlite3.connect(db)
    raw.executescript(
        """
        CREATE TABLE change_candidates (candidate_id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER, scene_a_id TEXT,
            scene_b_id TEXT, min_x REAL, min_y REAL, max_x REAL, max_y REAL, confidence REAL);
        """
    )
    raw.commit()
    raw.close()
    conn = init_connection(db)
    init_schema(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(change_candidates)")}
    conn.close()
    assert {"sub_blobs", "centroid_lon", "centroid_lat", "mgrs_ref", "geometry", "direction_evidence"} <= cols


# ---- end to end: the large-area pass and sorting by area ------------------------------------------------------------

# three patches with 40 px gaps: too far apart for the near pass, one large-area change for the wide pass
SCATTERED = [(100, 160, 60, 140), (100, 160, 180, 260), (100, 160, 300, 380)]
SMALL_AWAY = (400, 412, 400, 410)  # 12 x 10 = 120 px, well away from everything


def test_scattered_patches_across_a_large_area_become_one_detection_in_hectares(tmp_path):
    from fastapi.testclient import TestClient

    from main import app

    cands, job, geometry = run_site(tmp_path, SCATTERED)
    assert len(cands) == 1
    (c,) = cands
    assert c["sub_blobs"] == 3 and c["area_px"] == pytest.approx(3 * 4800, rel=0.05)
    assert c["direction_evidence"]["merged"]["large_area"] is True
    assert job["details"]["grouping"]["large_area_groups"] == 1 and job["details"]["grouping"]["detections"] == 1
    assert json.loads(geometry[c["candidate_id"]])["type"] == "MultiPolygon"  # the outline is still the three patches

    client = TestClient(app)
    feature = client.get("/api/export/changes").json()["features"][0]["properties"]
    assert feature["area_ha"] == pytest.approx(144.0, rel=0.05)  # 14,400 px at 100 m² each
    detail = client.get(f"/api/changes/{c['candidate_id']}").json()
    grouping = next(line["text"] for line in detail["processing_details"] if line["key"] == "grouping")
    assert grouping.startswith("Large-area change: merged from 3 change blobs less than 60 px apart")


def test_the_near_pass_alone_would_have_left_the_scattered_patches_apart(tmp_path, monkeypatch):
    monkeypatch.setattr(params, "LARGE_AREA_MIN_BLOBS", 10**6)
    monkeypatch.setattr(params, "LARGE_AREA_MIN_PIXELS", 10**12)
    cands, job, _ = run_site(tmp_path, SCATTERED)
    assert len(cands) == 3 and {c["sub_blobs"] for c in cands} == {1}  # what the 10 px pass alone produced
    assert job["details"]["grouping"]["large_area_groups"] == 0


def test_small_scattered_changes_stay_individual_candidates(tmp_path):
    tiny = [(100, 112, 100, 110), (100, 112, 150, 160)]  # two 120 px patches, 40 px apart: 240 px in all
    cands, job, _ = run_site(tmp_path, tiny)
    assert len(cands) == 2 and {c["sub_blobs"] for c in cands} == {1}
    assert all(c["area_px"] < 500 for c in cands) and job["details"]["grouping"]["large_area_groups"] == 0


def test_sorting_by_area_puts_the_largest_first_and_the_cap_keeps_it(tmp_path):
    from fastapi.testclient import TestClient

    from main import app

    run_site(tmp_path, SCATTERED + [SMALL_AWAY])
    client = TestClient(app)
    by_area = client.get("/api/changes", params={"sort": "area"}).json()["candidates"]
    assert len(by_area) == 2 and by_area[0]["area_px"] > 10 * by_area[1]["area_px"]
    assert by_area[0]["sub_blobs"] == 3

    # capped to one per pair, an area sort keeps the biggest change whatever its confidence
    capped = client.get("/api/changes", params={"sort": "area", "limit": 1}).json()
    assert [c["candidate_id"] for c in capped["candidates"]] == [by_area[0]["candidate_id"]]
    assert capped["matching"] == 2 and capped["shown"] == 1
    by_conf = client.get("/api/changes", params={"sort": "confidence", "limit": 1}).json()["candidates"]
    top_confidence = max(by_area, key=lambda c: c["confidence"])
    assert by_conf[0]["candidate_id"] == top_confidence["candidate_id"]  # the other sorts still cap by confidence


def test_the_storage_cap_never_drops_the_largest_detections(monkeypatch):
    from change_detection.scoring import finalize_candidates

    monkeypatch.setattr(params, "MAX_STORED_CANDIDATES", 10)
    monkeypatch.setattr(params, "STORED_AREA_RESERVE", 3)
    rows = [{"confidence": 0.95 - i * 0.01, "area_px": 100 + i} for i in range(20)]  # confident ones are small
    rows.append({"confidence": 0.35, "area_px": 900_000})  # a huge, weakly scored change
    rows.append({"confidence": 0.36, "area_px": 500_000})
    trace = {}
    kept = finalize_candidates(rows, trace)
    assert len(kept) == 10 and trace["dropped_over_cap"] == 12 and trace["candidates_found"] == 22
    assert {r["area_px"] for r in kept} >= {900_000, 500_000}  # both survive despite low confidence
    assert [r["confidence"] for r in kept] == sorted((r["confidence"] for r in kept), reverse=True)
    # within the cap nothing is reserved or dropped
    monkeypatch.setattr(params, "MAX_STORED_CANDIDATES", 100)
    assert len(finalize_candidates(rows, {})) == 22
