"""Tests for the Landsat Collection 2 loader, the generic (Bhuvan / LISS-III) loader, and change detection on both."""

import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog import changes as store
from catalog.database import init_connection
from change_detection import params, run_change_detection
from change_detection.trigger import plan_pairs
from ingestion.landsat import (
    build_landsat_cogs,
    decode_qa,
    inspect_landsat_product,
    open_landsat_product,
    parse_landsat_name,
    qa_to_scl,
    to_s2_dn,
)
from ingestion.loader import (
    band_roles,
    convert_generic_to_cogs,
    detect_sensor,
    heuristic_cloud_pct,
    infer_sensor_from_raster,
    inspect_raster,
)
from main import app
from s2_factory import (
    LANDSAT_CLEAR_QA,
    Scenario,
    ingest_generic_without_embedding,
    ingest_landsat_without_embedding,
    ingest_without_embedding,
    make_generic,
    make_landsat,
    make_safe,
    render,
)

L8_A = "LC08_L2SP_146040_20240110_20240120_02_T1"
L8_B = "LC08_L2SP_146040_20240214_20240224_02_T1"
L9_B = "LC09_L2SP_146040_20240214_20240224_02_T1"
BOX = (200, 260, 300, 380)  # vegetation -> bare


# ---- reading a scene ------------------------------------------------------------------------------------------------------------


def test_names_are_recognised_and_sensor_follows_the_mission():
    assert parse_landsat_name(L8_A) and parse_landsat_name(L9_B)
    assert parse_landsat_name("S2A_MSIL2A_20240110T050649_N0510_R019_T43PHM") is None
    assert parse_landsat_name("LE07_L2SP_146040_20240110_20240120_02_T1") is None  # only Landsat 8 and 9
    assert detect_sensor(L8_A + "_SR_B4.TIF") == "landsat8" and detect_sensor(L9_B) == "landsat9"


def test_a_scene_opens_from_its_folder_or_from_one_of_its_files(tmp_path):
    folder = make_landsat(tmp_path, L8_A, Scenario(date="2024-01-10"))
    from_folder = open_landsat_product(folder)
    from_file = open_landsat_product(folder / f"{L8_A}_SR_B4.TIF")
    from_parent = open_landsat_product(tmp_path)  # the folder that holds the scene folder
    for product in (from_folder, from_file, from_parent):
        assert product.scene_id == L8_A and product.sensor == "landsat8"
        assert product.acquisition_date == "2024-01-10"  # the acquisition date, not the processing date
        assert product.wrs_path_row == "146/040"
        assert set(product.band_paths) == {"B2", "B3", "B4", "B5"} and product.qa_path.name.endswith("QA_PIXEL.TIF")
    assert open_landsat_product(make_landsat(tmp_path, L9_B, Scenario(date="2024-02-14"))).sensor == "landsat9"


def test_something_that_is_not_landsat_is_left_for_the_other_loaders(tmp_path):
    safe = make_safe(tmp_path, "S2A_MSIL2A_20240110T050649_N0510_R019_T43PHM_20240110T090000", Scenario(date="2024-01-10"))
    assert open_landsat_product(safe) is None
    generic = make_generic(tmp_path / "plain.tif", np.ones((3, 32, 32), dtype=np.uint8))
    assert open_landsat_product(generic) is None
    assert open_landsat_product(tmp_path / "does_not_exist") is None


def test_an_incomplete_or_ambiguous_scene_is_refused_with_a_clear_message(tmp_path):
    folder = make_landsat(tmp_path / "a", L8_A, Scenario(date="2024-01-10"))
    (folder / f"{L8_A}_SR_B5.TIF").unlink()
    with pytest.raises(ValueError, match="SR_B5"):
        open_landsat_product(folder)

    make_landsat(tmp_path / "two", L8_A, Scenario(date="2024-01-10"))
    make_landsat(tmp_path / "two", L8_B, Scenario(date="2024-02-14"))
    with pytest.raises(ValueError, match="single scene"):
        open_landsat_product(tmp_path / "two")


# ---- QA_PIXEL and the reflectance scaling ---------------------------------------------------------------------------------------


def test_qa_bits_are_decoded_as_specified():
    qa = np.array([LANDSAT_CLEAR_QA, LANDSAT_CLEAR_QA | 1 << 3, LANDSAT_CLEAR_QA | 1 << 4, LANDSAT_CLEAR_QA | 1 << 5, 1], dtype=np.uint16)
    flags = decode_qa(qa)
    assert flags["cloud"].tolist() == [False, True, False, False, False]  # (qa >> 3) & 1
    assert flags["shadow"].tolist() == [False, False, True, False, False]  # (qa >> 4) & 1
    assert flags["snow"].tolist() == [False, False, False, True, False]  # (qa >> 5) & 1
    assert flags["fill"].tolist() == [False, False, False, False, True]


def test_qa_becomes_sentinel2_class_codes_so_the_existing_mask_logic_applies():
    qa = np.array([LANDSAT_CLEAR_QA, LANDSAT_CLEAR_QA | 1 << 3, LANDSAT_CLEAR_QA | 1 << 4, LANDSAT_CLEAR_QA | 1 << 5, LANDSAT_CLEAR_QA | 1 << 3 | 1 << 5], dtype=np.uint16)
    scl = qa_to_scl(qa, no_data=np.zeros(5, dtype=bool))
    assert scl.tolist() == [7, 9, 3, 11, 9]  # clear, cloud, shadow, snow, and cloud wins over snow
    assert qa_to_scl(qa, no_data=np.array([True, False, False, False, False]))[0] == 0  # no data wins over everything
    invalid = {9, 3}  # cloud OR shadow
    assert all((c in params.SCL_VALID_CLASSES) == (c not in invalid) for c in (7, 9, 3, 11))  # snow stays valid, flagged


def test_reflectance_is_rewritten_in_the_sentinel2_convention():
    # Landsat: reflectance = DN * 2.75e-5 - 0.2 ; Sentinel-2 convention: reflectance = (DN' - 1000) / 10000
    reflectance = np.array([0.0, 0.1, 0.4, 0.8])
    raw = np.rint((reflectance + 0.2) / 2.75e-5).astype(np.uint16)
    back = (to_s2_dn(raw).astype(np.float64) - 1000.0) / 10000.0
    assert back == pytest.approx(reflectance, abs=2e-4)
    assert to_s2_dn(np.array([0, 5000, 7273], dtype=np.uint16)).tolist()[0] == 0  # fill stays no data
    assert to_s2_dn(np.array([1], dtype=np.uint16))[0] == 1  # a valid, very dark pixel is floored at 1, never no data


# ---- the COGs ---------------------------------------------------------------------------------------------------------------------------


def test_the_three_rasters_are_built_on_the_scene_grid(tmp_path):
    scn = Scenario(date="2024-01-10", cloud=(20, 100, 20, 100), landcover=[(*BOX, "bare")], nodata_rows=32)
    folder = make_landsat(tmp_path / "src", L8_A, scn, snow=(300, 340, 300, 340))
    product = open_landsat_product(folder)
    stages = []
    outputs = build_landsat_cogs(product, on_stage=lambda stage, done, total: stages.append(stage))
    assert set(stages) == {"bands", "qa", "rgb", "cog"}

    with rasterio.open(outputs.display_cog) as rgb, rasterio.open(outputs.analysis_cog) as analysis, rasterio.open(outputs.scl_cog) as mask:
        assert (rgb.count, rgb.dtypes[0]) == (3, "uint8") and (analysis.count, analysis.dtypes[0]) == (4, "uint16")
        assert (mask.count, mask.dtypes[0]) == (1, "uint8")
        assert rgb.transform == analysis.transform == mask.transform and rgb.res == (30.0, 30.0)
        assert analysis.tags()["BOA_ADD_OFFSETS"] == "-1000.0,-1000.0,-1000.0,-1000.0"
        assert analysis.overviews(1) and mask.overviews(1)

        scl = mask.read(1)
        assert set(np.unique(scl)) == {0, 7, 9, 11}  # fill, clear, cloud, snow
        assert (scl[40:80, 40:80] == 9).all() and (scl[310:330, 310:330] == 11).all() and (scl[-16:] == 0).all()

        # a bare patch: Landsat -> Sentinel-2 convention reproduces the reflectance the scene was built from
        bands, _ = render(scn)
        expected = bands[3, 210:250, 310:370].astype(np.float64)
        got = analysis.read(4)[210:250, 310:370].astype(np.float64)
        assert np.abs(got - expected).max() < 25  # the round trip through Landsat's 16-bit scaling
        assert (rgb.read()[:, -16:] == 0).all()  # no data stays no data on the map

    total = 512 * 512
    data_px = total - 32 * 512
    assert outputs.cloud_pct == pytest.approx(100.0 * (80 * 80) / data_px, abs=0.05)
    assert outputs.snow_pct == pytest.approx(100.0 * (40 * 40) / data_px, abs=0.05)
    assert outputs.nodata_pct == pytest.approx(100.0 * 32 / 512, abs=0.05)


def test_metadata_names_sensor_grid_and_provenance(tmp_path):
    product = open_landsat_product(make_landsat(tmp_path, L9_B, Scenario(date="2024-02-14")))
    meta = inspect_landsat_product(product)
    assert (meta.sensor, meta.acquisition_date, meta.crs, meta.resolution) == ("landsat9", "2024-02-14", "EPSG:32643", (30.0, 30.0))
    assert len(meta.raw_checksum) == 64 and meta.bounds_wgs84[0] < meta.bounds_wgs84[2]


# ---- through the API ---------------------------------------------------------------------------------------------------------------------


def test_ingest_endpoint_takes_a_landsat_folder_end_to_end(tmp_path):
    client = TestClient(app)
    folder = make_landsat(tmp_path / "src", L8_A, Scenario(date="2024-01-10", cloud=(20, 100, 20, 100)))
    pid = "landsat-test-0001"
    resp = client.post("/api/ingest", json={"file_path": str(folder), "pipeline_id": pid})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["product_type"], body["sensor"], body["scene_id"]) == ("landsat_c2", "landsat8", L8_A)
    assert body["cloud_pct"] == pytest.approx(100.0 * 80 * 80 / (512 * 512), abs=0.1)
    assert body["analysis_cog_path"].endswith("_analysis.tif") and body["scl_path"].endswith("_scl.tif")

    pipeline = client.get(f"/api/pipeline/{pid}").json()
    keys = [step["key"] for step in pipeline["steps"]]
    assert keys[:5] == ["detect", "bands", "qa", "rgb", "cog"] and pipeline["state"] == "done"  # background embedding finished too
    assert client.get(f"/api/status/{L8_A}").json()["state"] == "ready"


def test_a_single_band_file_is_enough_to_find_the_scene(tmp_path):
    client = TestClient(app)
    folder = make_landsat(tmp_path / "src", L8_A, Scenario(date="2024-01-10"))
    resp = client.post("/api/ingest", json={"file_path": str(folder / f"{L8_A}_SR_B4.TIF")})
    assert resp.status_code == 200 and resp.json()["product_type"] == "landsat_c2"


def test_a_broken_landsat_scene_is_a_400_with_no_paths_in_it(tmp_path):
    client = TestClient(app)
    folder = make_landsat(tmp_path / "src", L8_A, Scenario(date="2024-01-10"))
    (folder / f"{L8_A}_QA_PIXEL.TIF").unlink()
    resp = client.post("/api/ingest", json={"file_path": str(folder)})
    assert resp.status_code == 400 and "QA_PIXEL" in resp.json()["detail"] and str(tmp_path) not in resp.json()["detail"]


# ---- change detection on Landsat ---------------------------------------------------------------------------------------------------------


def landsat_pair(tmp_path, b_name=L8_B, **b_kwargs):
    a = ingest_landsat_without_embedding(
        make_landsat(tmp_path / "src", L8_A, Scenario(date="2024-01-10", noise_seed=1, landcover=[(*BOX, "vegetation")]))
    )
    b_scn = Scenario(date="2024-02-14", noise_seed=2, landcover=[(*BOX, "bare")], **b_kwargs)
    b = ingest_landsat_without_embedding(make_landsat(tmp_path / "src", b_name, b_scn))
    return a, b


def test_change_detection_finds_the_change_and_masks_qa_clouds(tmp_path):
    a, b = landsat_pair(tmp_path, cloud=(20, 120, 20, 120))  # a cloud the QA band flags in the newer scene
    result = run_change_detection(a, b)
    assert result["status"] == "completed", result

    conn = init_connection()
    try:
        (cand,) = store.list_candidates(conn)  # the change, and nothing at the cloud
        details = store.list_jobs(conn)[0]["details"]
    finally:
        conn.close()
    assert (cand["change_type"], cand["direction"]) == ("clearance", "disappearance")
    assert cand["scene_a_date"] == "2024-01-10" and cand["sensor"] == "landsat8"
    # area is in hectares from the 30 m pixel size: 4,800 px x 0.09 ha, not the 48 ha a 10 m pixel would give
    assert cand["area_ha"] == pytest.approx(cand["area_px"] * 0.09, rel=1e-3) and cand["area_ha"] > 300
    assert details["masking"]["scene_b"]["source"] == "qa_pixel" and details["masking"]["scene_b"]["cloud_trust"] < 0.97
    assert details["masking"]["scene_a"]["source"] == "qa_pixel" and details["masking"]["scene_a"]["cloud_trust"] == 1.0
    assert cand["min_lon"] < cand["max_lon"]


def test_the_trace_names_the_qa_mask(tmp_path):
    a, b = landsat_pair(tmp_path)
    run_change_detection(a, b)
    client = TestClient(app)
    cid = client.get("/api/changes").json()["candidates"][0]["candidate_id"]
    lines = {line["key"]: line["text"] for line in client.get(f"/api/changes/{cid}").json()["processing_details"]}
    assert lines["masking"].startswith("QA_PIXEL-based, CloudTrust A: 1.00")
    export = client.get("/api/export/changes").json()["features"][0]["properties"]
    assert export["processing"]["masking"] == "QA_PIXEL" and export["sensor"] == "landsat8"
    assert export["area_ha"] == pytest.approx(export["area_px"] * 0.09, rel=1e-2)


def test_landsat_never_pairs_with_sentinel_2_or_the_other_landsat(tmp_path):
    l8 = ingest_landsat_without_embedding(make_landsat(tmp_path / "src", L8_A, Scenario(date="2024-01-10")))
    s2_name = "S2A_MSIL2A_20240214T050649_N0510_R019_T43PHM_20240214T090000"
    s2 = ingest_without_embedding(make_safe(tmp_path / "src", s2_name, Scenario(date="2024-02-14")))
    l9 = ingest_landsat_without_embedding(make_landsat(tmp_path / "src", L9_B, Scenario(date="2024-02-14")))

    assert plan_pairs(s2) == [] and plan_pairs(l9) == []  # same ground, different sensors: no comparison is planned
    with pytest.raises(ValueError, match="same sensor"):
        run_change_detection(l8, s2)
    with pytest.raises(ValueError, match="same sensor"):
        run_change_detection(l8, l9)


def test_two_scenes_of_one_landsat_do_pair_automatically(tmp_path):
    a, b = landsat_pair(tmp_path)
    assert plan_pairs(b) == [(a, b)]


# ---- generic rasters ------------------------------------------------------------------------------------------------------------------------


def s2_like(day="2024-01-10", **kwargs):
    """(4, H, W) uint16 blue/green/red/NIR for a scenario, without clouds unless asked for."""
    return render(Scenario(date=day, **kwargs))[0]


def test_sensor_is_inferred_from_bands_and_resolution():
    assert infer_sensor_from_raster(4, (23.5, 23.5), projected=True) == "liss3"
    assert infer_sensor_from_raster(3, (23.5, 23.5), projected=True) == "liss3"
    assert infer_sensor_from_raster(4, (30.0, 30.0), projected=True) == "unknown"
    assert infer_sensor_from_raster(3, (10.0, 10.0), projected=True) == "unknown"
    assert infer_sensor_from_raster(4, (0.000212, 0.000212), projected=False) == "unknown"  # degrees say nothing
    assert infer_sensor_from_raster(5, (23.5, 23.5), projected=True) == "unknown"


def test_a_raster_is_typed_from_its_bands_when_its_name_says_nothing(tmp_path):
    liss = make_generic(tmp_path / "scene_2024-01-10.tif", s2_like()[:4], res=23.5)
    assert inspect_raster(liss).sensor == "liss3"
    rgb = make_generic(tmp_path / "photo_2024-01-10.tif", s2_like()[:3] // 40, res=10.0)
    assert inspect_raster(rgb).sensor == "unknown"
    named = make_generic(tmp_path / "LISS3_2024-01-10.tif", s2_like()[:3] // 40, res=10.0)
    assert inspect_raster(named).sensor == "liss3"  # the name says so


def test_band_roles_follow_the_sensor():
    assert band_roles("liss3", 4) == {"blue": None, "green": 0, "red": 1, "nir": 2}  # green, red, NIR, SWIR
    assert band_roles("unknown", 4) == {"blue": 0, "green": 1, "red": 2, "nir": 3}
    assert band_roles("unknown", 5)["nir"] == 3
    assert band_roles("unknown", 3) == {"blue": None, "green": None, "red": None, "nir": None}


def test_rgb_only_gets_no_masking_and_says_so(tmp_path, caplog):
    path = make_generic(tmp_path / "photo_2024-01-10.tif", (s2_like()[:3] // 40).astype(np.uint8), res=10.0)
    with caplog.at_level(logging.WARNING, logger="iris.ingestion"):
        assert heuristic_cloud_pct(path, "unknown") is None
    assert "no cloud masking is possible, CloudTrust = 1.0" in caplog.text


def test_a_bright_unvegetated_patch_is_flagged_as_possibly_cloud_and_logged(tmp_path, caplog):
    data = s2_like()
    data[:, 40:200, 40:200] = np.array([4000, 4200, 4300, 3900], dtype=np.uint16)[:, None, None]  # bright, NDVI < 0
    path = make_generic(tmp_path / "unnamed_2024-01-10.tif", data, res=10.0)
    with caplog.at_level(logging.INFO, logger="iris.ingestion.loader"):
        pct = heuristic_cloud_pct(path, "unknown")
    assert pct is not None and 8.0 < pct < 16.0  # the 160 x 160 patch is about 9.8% of the scene
    assert "Heuristic masking applied — no QA band" in caplog.text


def test_a_clear_scene_flags_only_a_little(tmp_path):
    # Vegetated everywhere, but the texture puts a percent or so of pixels both bright and low in NDVI: the price of a
    # heuristic that has no QA band to check against
    path = make_generic(tmp_path / "unnamed_2024-01-10.tif", s2_like(), res=10.0)
    assert 0.0 <= heuristic_cloud_pct(path, "unknown") < 3.0


def test_a_liss3_raster_gets_an_rgb_display_and_an_analysis_cog_in_pipeline_order(tmp_path):
    rgbn = s2_like()  # blue, green, red, NIR
    liss = np.stack([rgbn[1], rgbn[2], rgbn[3], rgbn[0]])  # green, red, NIR, SWIR (any values)
    path = make_generic(tmp_path / "LISS3_2024-01-10.tif", liss, res=23.5)
    meta = inspect_raster(path)
    out = convert_generic_to_cogs(path, meta)
    assert out.analysis_cog is not None
    with rasterio.open(out.display_cog) as display, rasterio.open(out.analysis_cog) as analysis:
        assert (display.count, display.dtypes[0]) == (3, "uint8") and analysis.count == 4
        got = analysis.read()
        assert (got[1] == rgbn[1]).all() and (got[2] == rgbn[2]).all() and (got[3] == rgbn[3]).all()  # green, red, NIR
        assert (got[0] == rgbn[1]).all()  # a LISS-III has no blue: green stands in
        assert "green" in analysis.tags()["NOTE"]
        assert analysis.tags()["BOA_ADD_OFFSETS"] == "0,0,0,0"
        shown = display.read()
        assert shown.min() == 0 or shown.min() >= 1
        assert not (shown[:, 10:200, 10:200] == 0).all()


def test_a_three_band_raster_stays_a_single_plain_cog(tmp_path):
    path = make_generic(tmp_path / "photo_2024-01-10.tif", (s2_like()[:3] // 40).astype(np.uint8), res=10.0)
    out = convert_generic_to_cogs(path, inspect_raster(path))
    assert out.analysis_cog is None
    with rasterio.open(out.display_cog) as cog:
        assert cog.count == 3 and cog.overviews(1)


def generic_pair(tmp_path, cloudy: bool):
    a_data = s2_like("2024-01-10", noise_seed=1, landcover=[(*BOX, "vegetation")])
    b_data = s2_like("2024-02-14", noise_seed=2, landcover=[(*BOX, "bare")])
    if cloudy:
        b_data[:, 40:200, 40:200] = np.array([4000, 4200, 4300, 3900], dtype=np.uint16)[:, None, None]
    a = ingest_generic_without_embedding(make_generic(tmp_path / "src" / "unit_2024-01-10.tif", a_data, res=10.0))
    b = ingest_generic_without_embedding(make_generic(tmp_path / "src" / "unit_2024-02-14.tif", b_data, res=10.0))
    return a, b


def test_heuristic_masking_lowers_confidence_but_removes_no_pixel(tmp_path):
    a, b = generic_pair(tmp_path, cloudy=True)
    conn = init_connection()
    try:
        assert conn.execute("SELECT cloud_pct FROM scenes WHERE scene_id = ?", (a,)).fetchone()[0] < 3.0
        assert 5.0 < conn.execute("SELECT cloud_pct FROM scenes WHERE scene_id = ?", (b,)).fetchone()[0] < 20.0
        assert conn.execute("SELECT scl_path FROM scenes WHERE scene_id = ?", (b,)).fetchone()[0] is None  # no mask exists
    finally:
        conn.close()

    assert run_change_detection(a, b)["status"] == "completed"
    conn = init_connection()
    try:
        job = store.list_jobs(conn)[0]
        cands = store.list_candidates(conn)
    finally:
        conn.close()
    masking = job["details"]["masking"]
    assert masking["scene_b"]["source"] == "heuristic" and 0.8 < masking["scene_b"]["cloud_trust"] < 0.95
    assert masking["scene_a"]["source"] == "heuristic" and masking["scene_a"]["cloud_trust"] > 0.97
    assert job["details"]["mutual_coverage"] > 0.99  # nothing was removed: every data pixel is still valid
    assert job["details"]["scoring"]["cloud_trust_factor"] == pytest.approx(masking["scene_b"]["cloud_trust"], abs=1e-3)
    # the cleared box is found (the flagged patch is not removed, so it may be found too: that is the cost of a heuristic)
    assert any((c["change_type"], c["direction"]) == ("clearance", "disappearance") for c in cands)


def test_the_heuristic_estimate_reaches_confidence_through_the_coverage_term(tmp_path):
    a, b = generic_pair(tmp_path, cloudy=True)
    run_change_detection(a, b)
    conn = init_connection()
    try:
        job = store.list_jobs(conn)[0]
        cands = store.list_candidates(conn)
    finally:
        conn.close()
    trust = job["details"]["scoring"]["cloud_trust_factor"]
    assert trust < 1.0
    for c in cands:
        # the stored coverage term is the observed coverage times CloudTrust, so the breakdown still sums to the score
        assert c["valid_coverage"] <= trust + 1e-6
        w = params.CONFIDENCE_WEIGHTS
        raw = w["alignment_quality"] * c["alignment_quality"] + w["cluster_distance"] * c["norm_cluster_dist"] + w["terrain_flatness"] * c["terrain_flatness"] + w["valid_coverage"] * c["valid_coverage"]
        factor = (c["direction_evidence"] or {}).get("seasonality", {}).get("confidence_factor", 1.0)
        assert c["confidence"] == pytest.approx(raw * factor, abs=1e-6)

    client = TestClient(app)
    cid = client.get("/api/changes").json()["candidates"][0]["candidate_id"]
    lines = {line["key"]: line["text"] for line in client.get(f"/api/changes/{cid}").json()["processing_details"]}
    assert lines["masking"].startswith("Heuristic (no QA band")


def test_a_clean_generic_pair_keeps_full_trust(tmp_path):
    a, b = generic_pair(tmp_path, cloudy=False)
    run_change_detection(a, b)
    conn = init_connection()
    try:
        details = store.list_jobs(conn)[0]["details"]
    finally:
        conn.close()
    assert details["scoring"]["cloud_trust_factor"] > 0.95
