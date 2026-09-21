"""Tests for Part B/C: the change-detection pipeline (Phases 1-5) and its trigger."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import change_detection.alignment as alignment_module
from catalog import changes
from catalog.database import init_connection, init_schema, upsert_scene
from change_detection import params, run_change_detection
from change_detection.trigger import plan_pairs, reconcile_interrupted_jobs
from s2_factory import Scenario, ingest_without_embedding, make_safe, render

NAME_A = "S2A_MSIL2A_20240110T050649_N0510_R019_T43PHM_20240110T090000"
NAME_B = "S2A_MSIL2A_20240214T050649_N0510_R019_T43PHM_20240214T090000"
# Same footprint but a different MGRS tile ID: alignment is only skipped for scenes of the SAME tile, so tests that
# exercise ECC/ORB must not share one
NAME_B_OTHER_TILE = "S2A_MSIL2A_20240214T050649_N0510_R019_T43PHN_20240214T090000"

# rows 200-260, cols 300-380 of a 512x512 scene whose upper-left corner is (300000, 3100000) at 10 m
PATCH = (200, 260, 300, 380, 0.35)  # NIR drops to 35% (red rises): a change, but one no land-cover class explains
CLEARED = (200, 260, 300, 380)  # vegetation on the first date, bare ground on the second
PATCH_NATIVE = (303000.0, 3097400.0, 303800.0, 3098000.0)  # min_x, min_y, max_x, max_y


def _job_and_candidates(a_id: str, b_id: str):
    conn = init_connection()
    try:
        init_schema(conn)
        return changes.get_job_for_pair(conn, a_id, b_id), changes.list_candidates(conn)
    finally:
        conn.close()


def run_pair(tmp_path, scene_a: Scenario, scene_b: Scenario, name_b: str = NAME_B):
    a_id = ingest_without_embedding(make_safe(tmp_path / "src", NAME_A, scene_a))
    b_id = ingest_without_embedding(make_safe(tmp_path / "src", name_b, scene_b))
    result = run_change_detection(a_id, b_id)
    job, cands = _job_and_candidates(a_id, b_id)
    return a_id, b_id, result, job, cands


def base_a() -> Scenario:
    return Scenario(date="2024-01-10", noise_seed=1)


def test_real_change_is_found_where_it_happened_despite_shift_and_gain(tmp_path):
    scene_a = Scenario(date="2024-01-10", noise_seed=1, landcover=[(*CLEARED, "vegetation")])
    scene_b = Scenario(
        date="2024-02-14", noise_seed=2, gain=1.06, offset=40, landcover=[(*CLEARED, "bare")], shift=(2.6, -1.4)
    )
    _, _, result, job, cands = run_pair(tmp_path, scene_a, scene_b, name_b=NAME_B_OTHER_TILE)

    assert result["status"] == "completed"
    assert len(cands) == 1
    cand = cands[0]

    # located on the patch (within the 1 px morphology/alignment tolerance)
    assert cand["min_x"] == pytest.approx(PATCH_NATIVE[0], abs=30)
    assert cand["min_y"] == pytest.approx(PATCH_NATIVE[1], abs=30)
    assert cand["max_x"] == pytest.approx(PATCH_NATIVE[2], abs=30)
    assert cand["max_y"] == pytest.approx(PATCH_NATIVE[3], abs=30)
    assert 0.9 * 4800 <= cand["area_px"] <= 1.05 * 4800
    assert cand["direction"] == "disappearance" and cand["change_type"] == "clearance"
    assert cand["mean_dndvi"] < -0.15
    assert -180 <= cand["min_lon"] < cand["max_lon"] <= 180 and -90 <= cand["min_lat"] < cand["max_lat"] <= 90

    # all four confidence terms are in [0, 1] and combine with the specified weights
    assert 0 < cand["confidence"] <= 1
    expected = (
        0.35 * cand["alignment_quality"]
        + 0.35 * cand["norm_cluster_dist"]
        + 0.15 * cand["terrain_flatness"]
        + 0.15 * cand["valid_coverage"]
    )
    # a vegetation clearance with no same-season history is "unverified", which discounts it (seasonal.py)
    assert cand["seasonality_status"] == "unverified"
    assert cand["confidence"] == pytest.approx(expected * params.UNVERIFIED_CONFIDENCE_FACTOR, abs=1e-9)
    assert cand["terrain_flatness"] == 1.0  # documented placeholder: no DEM yet

    # decision trace: ECC recovered the injected 2.6/-1.4 px shift, and PIF normalisation ran
    align = job["details"]["alignment"]
    assert align["method"] == "ecc" and align["rho"] > 0.9 and align["warp_applied"]
    assert align["matrix"][0][2] == pytest.approx(-2.6, abs=0.15)
    assert align["matrix"][1][2] == pytest.approx(1.4, abs=0.15)
    assert job["details"]["radiometry"]["tier"] == "pif"
    assert job["details"]["detection"]["kmeans_centroids"][0] < job["details"]["detection"]["kmeans_centroids"][1]
    assert {"masking", "grid_check", "alignment", "radiometry", "detection", "scoring", "total"} <= set(job["details"]["timings_s"])
    assert job["status"] == "completed"


def test_no_change_means_no_candidates_even_with_shift_and_gain(tmp_path):
    scene_b = Scenario(date="2024-02-14", noise_seed=2, gain=1.06, offset=40, shift=(2.6, -1.4))
    _, _, result, job, cands = run_pair(tmp_path, base_a(), scene_b, name_b=NAME_B_OTHER_TILE)
    assert result["status"] == "completed"
    assert cands == []

    # Normalisation must recover the injected radiometric difference: B * (1/1.06) maps B back onto A.
    # Selecting PIFs on |A - B| alone fitted 1.05 for a true 0.952 on a real scene (it hides the gain), so pin it.
    radiometry = job["details"]["radiometry"]
    assert radiometry["tier"] == "pif"
    for slope in radiometry["slope"]:
        assert slope == pytest.approx(1 / 1.06, abs=0.03)


def test_radiometric_gain_is_recovered_in_the_direction_that_corrects_it(tmp_path):
    """B is brighter than A (gain 1.2): the fitted gain must be below 1, not above."""
    scene_b = Scenario(date="2024-02-14", noise_seed=2, gain=1.2, offset=0.0)
    _, _, _, job, _ = run_pair(tmp_path, base_a(), scene_b)
    for slope in job["details"]["radiometry"]["slope"]:
        assert slope == pytest.approx(1 / 1.2, abs=0.04)


def test_clouds_are_masked_so_they_never_become_change(tmp_path):
    """A bright cloud in B is a huge apparent difference; SCL masking must keep it out of the result."""
    scene_b = Scenario(date="2024-02-14", noise_seed=2, cloud=(100, 300, 100, 300))
    _, _, result, job, cands = run_pair(tmp_path, base_a(), scene_b)

    assert result["status"] == "completed"
    assert cands == []
    masking = job["details"]["masking"]
    assert masking["scene_a"]["cloud_trust"] > 0.99
    assert masking["scene_b"]["cloud_trust"] == pytest.approx(1 - 200 * 200 / 512**2, abs=0.01)
    assert job["details"]["mutual_coverage"] < 0.9


def test_change_under_cloud_is_not_reported(tmp_path):
    scene_b = Scenario(date="2024-02-14", noise_seed=2, patches=[PATCH], cloud=(180, 280, 280, 400))
    _, _, _, _, cands = run_pair(tmp_path, base_a(), scene_b)
    assert cands == []  # nothing was observed there, so nothing may be claimed


def test_different_pixel_grid_is_warped_onto_a_then_differenced(tmp_path):
    # same CRS, but the pixel origin is offset by a non-integer number of pixels
    scene_b = Scenario(date="2024-02-14", noise_seed=2, patches=[PATCH], origin=(300000.0 + 37.5, 3100000.0 - 22.5))
    _, _, result, job, cands = run_pair(tmp_path, base_a(), scene_b)

    assert job["details"]["grid"] == {"matched": False, "regridded": True, "scl_resampling": "nearest"}
    assert result["status"] == "completed"
    assert len(cands) == 1
    assert cands[0]["min_x"] == pytest.approx(PATCH_NATIVE[0], abs=40)


def test_insufficient_mutual_coverage_is_not_differenced(tmp_path):
    scene_b = Scenario(date="2024-02-14", noise_seed=2, nodata_rows=400)  # 78% of B has no data
    _, _, result, job, cands = run_pair(tmp_path, base_a(), scene_b)

    assert result["status"] == "insufficient_evidence"
    assert job["status"] == "insufficient_evidence"
    assert "30%" in job["error_message"]
    assert cands == []
    assert "alignment" not in job["details"]  # the pipeline stopped before aligning


def test_ecc_failure_falls_back_to_orb_ransac(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise cv2.error("NaN encountered.")  # what ECC raises on a NaN pixel

    monkeypatch.setattr(cv2, "findTransformECC", boom)
    scene_b = Scenario(date="2024-02-14", noise_seed=2, gain=1.06, offset=40, patches=[PATCH], shift=(2.6, -1.4))
    _, _, result, job, cands = run_pair(tmp_path, base_a(), scene_b, name_b=NAME_B_OTHER_TILE)

    align = job["details"]["alignment"]
    assert align["method"] == "orb"
    assert align["inlier_ratio"] > 0.3 and align["rho"] is None
    assert "NaN encountered" in align["ecc_failure"]
    assert result["status"] == "completed" and len(cands) == 1
    assert cands[0]["alignment_quality"] == pytest.approx(align["inlier_ratio"], abs=1e-3)  # inlier ratio, not rho


def test_low_rho_is_a_failure_even_though_ecc_did_not_raise(tmp_path, monkeypatch):
    real = cv2.findTransformECC

    def converge_badly(*args):
        _, warp = real(*args)
        return 0.42, warp  # ECC can return a poor transform without any exception

    monkeypatch.setattr(cv2, "findTransformECC", converge_badly)
    scene_b = Scenario(date="2024-02-14", noise_seed=2, patches=[PATCH], shift=(2.6, -1.4))
    _, _, _, job, _ = run_pair(tmp_path, base_a(), scene_b, name_b=NAME_B_OTHER_TILE)

    assert job["details"]["alignment"]["method"] == "orb"
    assert "rho 0.420 below 0.8" in job["details"]["alignment"]["ecc_failure"]


def test_alignment_failed_when_neither_method_passes(tmp_path):
    """Unrelated ground: ECC does not converge, ORB finds no consistent geometry."""
    scene_b = Scenario(date="2024-02-14", noise_seed=2, seed=99)
    _, _, result, job, cands = run_pair(tmp_path, base_a(), scene_b, name_b=NAME_B_OTHER_TILE)

    assert result["status"] == "alignment_failed"
    assert job["status"] == "alignment_failed"
    assert cands == []
    assert "ORB fallback also failed" in job["error_message"]
    # nothing that identifies the build machine or the file system may reach the analyst
    assert ":\\" not in job["error_message"] and "opencv-python" not in job["error_message"]


def test_only_same_sensor_scenes_are_differenced(tmp_path):
    a_id = ingest_without_embedding(make_safe(tmp_path / "src", NAME_A, base_a()))
    b_id = ingest_without_embedding(make_safe(tmp_path / "src", NAME_B, Scenario(date="2024-02-14", noise_seed=2)))
    conn = init_connection()
    conn.execute("UPDATE scenes SET sensor='landsat8' WHERE scene_id=?", (b_id,))
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="same sensor"):
        run_change_detection(a_id, b_id)
    assert plan_pairs(b_id) == []  # the trigger never even pairs them


def test_pair_is_ordered_by_date_and_never_processed_twice(tmp_path):
    a_id = ingest_without_embedding(make_safe(tmp_path / "src", NAME_A, base_a()))
    b_id = ingest_without_embedding(
        make_safe(tmp_path / "src", NAME_B, Scenario(date="2024-02-14", noise_seed=2, patches=[PATCH]))
    )

    first = run_change_detection(b_id, a_id)  # arguments deliberately reversed: A must still be the older scene
    job, cands = _job_and_candidates(a_id, b_id)
    assert job["scene_a_id"] == a_id and job["scene_b_id"] == b_id
    assert first["status"] == "completed" and len(cands) == 1

    second = run_change_detection(a_id, b_id)
    _, cands_after = _job_and_candidates(a_id, b_id)
    assert second["job_id"] == first["job_id"] and second["status"] == "completed"
    assert len(cands_after) == 1  # UNIQUE(job_type, scene_a_id, scene_b_id): no duplicate job, no duplicate rows

    conn = init_connection()
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    conn.close()


def test_scenes_need_different_known_dates(tmp_path):
    a_id = ingest_without_embedding(make_safe(tmp_path / "src", NAME_A, base_a()))
    b_id = ingest_without_embedding(make_safe(tmp_path / "src", NAME_B, Scenario(date="2024-01-10", noise_seed=2)))
    with pytest.raises(ValueError, match="different, known acquisition dates"):
        run_change_detection(a_id, b_id)
    assert plan_pairs(b_id) == []  # same date = an adjacent tile of one pass, not a repeat observation


def test_trigger_pairs_with_nearest_prior_and_nearest_later_scene(tmp_path):
    def add(name, date, **kw):
        return ingest_without_embedding(make_safe(tmp_path / "src", name, Scenario(date=date, **kw)))

    jan = add("S2A_MSIL2A_20240110T050649_x_T43PHM_a", "2024-01-10")
    assert plan_pairs(jan) == []  # nothing else catalogued yet

    mar = add("S2A_MSIL2A_20240310T050649_x_T43PHM_b", "2024-03-10")
    assert plan_pairs(mar) == [(jan, mar)]  # Jan is Mar's only (hence nearest) prior observation

    feb = add("S2A_MSIL2A_20240214T050649_x_T43PHM_c", "2024-02-14")
    # Feb sits between them: nearest prior = Jan, nearest later = Mar. Never Jan->Mar again for Feb's sake, and
    # never "every overlapping scene", which grows combinatorially as the archive fills.
    assert plan_pairs(feb) == [(jan, feb), (feb, mar)]

    conn = init_connection()
    rows = conn.execute("SELECT scene_a_id, scene_b_id, status FROM jobs ORDER BY job_id").fetchall()
    conn.close()
    assert [(a, b) for a, b, _ in rows] == [(jan, mar), (jan, feb), (feb, mar)]
    assert all(s == "queued" for _, _, s in rows)

    plan_pairs(feb)  # idempotent: UNIQUE constraint, no duplicate jobs
    conn = init_connection()
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 3
    conn.close()


def test_non_overlapping_scenes_are_not_paired(tmp_path):
    a_id = ingest_without_embedding(make_safe(tmp_path / "src", NAME_A, base_a()))
    far = ingest_without_embedding(
        make_safe(tmp_path / "src", NAME_B, Scenario(date="2024-02-14", origin=(400000.0, 3300000.0)))
    )
    assert plan_pairs(far) == []


def test_interrupted_jobs_are_requeued_on_startup(tmp_path):
    a_id = ingest_without_embedding(make_safe(tmp_path / "src", NAME_A, base_a()))
    b_id = ingest_without_embedding(make_safe(tmp_path / "src", NAME_B, Scenario(date="2024-02-14")))
    conn = init_connection()
    job = changes.create_job(conn, a_id, b_id)
    changes.mark_job_processing(conn, job["job_id"])  # the app was closed mid-job
    conn.close()

    assert reconcile_interrupted_jobs() == 1
    conn = init_connection()
    assert changes.get_job_for_pair(conn, a_id, b_id)["status"] == "queued"
    conn.close()


def _plain_geotiff(path: Path, scene: Scenario) -> None:
    """A non-Sentinel-2-SAFE input: a 3-band 8-bit GeoTIFF with no SCL."""
    bands, _ = render(scene)
    rgb = np.clip(bands[:3].astype(np.float32) / 12.0, 1, 255).astype(np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", height=scene.size, width=scene.size, count=3, dtype="uint8", crs=scene.crs,
        transform=from_origin(scene.origin[0], scene.origin[1], 10.0, 10.0), nodata=0,
    ) as dst:
        dst.write(rgb)


def test_scene_without_scl_is_all_valid_and_still_processed(tmp_path):
    """Non-Sentinel-2 input: no SCL, so every data pixel is valid, CloudTrust=1.0 and a warning is logged."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    a_file = tmp_path / "S2A_MSIL2A_20240110T050649_plain.tif"
    b_file = tmp_path / "S2A_MSIL2A_20240214T050649_plain.tif"
    _plain_geotiff(a_file, base_a())
    _plain_geotiff(b_file, Scenario(date="2024-02-14", noise_seed=2, patches=[PATCH], gain=1.04))

    assert client.post("/api/ingest", json={"file_path": str(a_file)}).status_code == 200
    ingest_b = client.post("/api/ingest", json={"file_path": str(b_file)})
    assert ingest_b.status_code == 200

    # the trigger ran after the second scene's embedding job
    conn = init_connection()
    a_id, b_id = "S2A_MSIL2A_20240110T050649_plain", "S2A_MSIL2A_20240214T050649_plain"
    job = changes.get_job_for_pair(conn, a_id, b_id)
    cands = changes.list_candidates(conn)
    conn.close()

    assert job is not None and job["status"] == "completed", job
    assert job["details"]["masking"]["scene_a"] == {"source": "data_mask", "cloud_trust": 1.0, "snow_fraction": 0.0}
    assert job["details"]["detection"]["difference_band"].startswith("mean intensity")
    assert len(cands) >= 1
    assert cands[0]["change_type"] == "unclassified"  # no NIR band, so no NDVI naming


# ---------------------------------------------------------------------------- same-tile alignment skip


@pytest.mark.parametrize(
    "scene_id, expected",
    [
        ("S2A_MSIL2A_20220623T052701_N0510_R105_T43RGM_20220623T090000", "43RGM"),
        ("T43PHM_20260909T050649_TCI_10m", "43PHM"),
        ("S2B_MSIL2A_20260909T050649_N0512_R019_T43PHM_20260909T085145", "43PHM"),
        ("S2A_MSIL2A_20240110T050649_plain", None),  # no tile in the name
        ("foo_T050649_bar", None),  # a time of day is not a tile
        ("", None),
    ],
)
def test_mgrs_tile_is_parsed_from_the_scene_id(scene_id, expected):
    from change_detection.rasters import mgrs_tile_id

    assert mgrs_tile_id(scene_id) == expected


def _forbid_alignment(monkeypatch):
    """Fail loudly if any intensity-based alignment runs."""

    def boom(*args, **kwargs):
        raise AssertionError("ECC/ORB must not run for a same-tile pair")

    monkeypatch.setattr(cv2, "findTransformECC", boom)
    monkeypatch.setattr(alignment_module, "estimate_alignment", boom)
    import change_detection.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "estimate_alignment", boom)


def test_same_tile_pair_skips_ecc_and_orb_entirely(tmp_path, monkeypatch, caplog):
    _forbid_alignment(monkeypatch)
    scene_b = Scenario(date="2024-02-14", noise_seed=2, gain=1.06, offset=40, patches=[PATCH])
    with caplog.at_level("INFO", logger="iris.change.alignment"), caplog.at_level("INFO", logger="iris.change.pipeline"):
        _, _, result, job, cands = run_pair(tmp_path, base_a(), scene_b)  # both names carry T43PHM

    assert result["status"] == "completed" and len(cands) == 1
    align = job["details"]["alignment"]
    assert align["method"] == "same_tile"
    assert align["warp_applied"] is False and align["matrix"] is None
    # good but not measured, so it must not outrank a real ECC correlation
    assert align["quality"] == 0.9 and align["rho"] is None and align["inlier_ratio"] is None
    assert "Same MGRS tile 43PHM" in align["note"]
    assert align["tiles"] == ["43PHM", "43PHM"]
    assert job["details"]["grid"] == {"matched": True}  # the grid check still ran and gated the skip
    assert "Same MGRS tile, grid alignment verified by definition" in caplog.text
    assert cands[0]["alignment_quality"] == 0.9


def test_same_tile_pair_with_large_real_change_is_not_rejected_as_misalignment(tmp_path):
    """Intensity-based alignment reads big land-cover change as misregistration; same-tile pairs must not."""
    big = [(0, 512, 200, 512, 0.3), (150, 400, 0, 200, 2.2)]  # about 60% of the scene changes
    scene_b = Scenario(date="2024-02-14", noise_seed=2, patches=big)
    _, _, result, job, cands = run_pair(tmp_path, base_a(), scene_b)

    assert result["status"] == "completed", job["error_message"]
    assert job["details"]["alignment"]["method"] == "same_tile"
    assert cands  # the change is reported, not discarded


def test_different_tiles_still_run_intensity_based_alignment(tmp_path):
    scene_b = Scenario(date="2024-02-14", noise_seed=2, patches=[PATCH], shift=(1.5, -0.8))
    _, _, result, job, _ = run_pair(tmp_path, base_a(), scene_b, name_b=NAME_B_OTHER_TILE)

    assert result["status"] == "completed"
    assert job["details"]["alignment"]["method"] == "ecc"
    assert job["details"]["alignment"]["tiles"] == ["43PHM", "43PHN"]


def test_unparseable_tile_ids_still_run_alignment(tmp_path):
    a_file = tmp_path / "S2A_MSIL2A_20240110T050649_plain.tif"
    b_file = tmp_path / "S2A_MSIL2A_20240214T050649_plain.tif"
    _plain_geotiff(a_file, base_a())
    _plain_geotiff(b_file, Scenario(date="2024-02-14", noise_seed=2, patches=[PATCH], shift=(1.5, -0.8)))
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    client.post("/api/ingest", json={"file_path": str(a_file)})
    client.post("/api/ingest", json={"file_path": str(b_file)})
    conn = init_connection()
    job = changes.get_job_for_pair(conn, "S2A_MSIL2A_20240110T050649_plain", "S2A_MSIL2A_20240214T050649_plain")
    conn.close()
    assert job["details"]["alignment"]["method"] in ("ecc", "orb")
    assert job["details"]["alignment"]["tiles"] == [None, None]


def test_same_tile_ids_but_a_different_grid_are_still_aligned(tmp_path, monkeypatch):
    """The tile ID is only trusted when the grid check confirms it; a 3.75 px origin offset must not be skipped."""
    scene_b = Scenario(date="2024-02-14", noise_seed=2, patches=[PATCH], origin=(300000.0 + 37.5, 3100000.0 - 22.5))
    _, _, result, job, _ = run_pair(tmp_path, base_a(), scene_b)  # same tile name, different grid

    assert job["details"]["grid"]["matched"] is False
    assert job["details"]["alignment"]["method"] in ("ecc", "orb")
    assert result["status"] == "completed"
