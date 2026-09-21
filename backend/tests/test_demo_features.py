"""Tests for the demo-readiness features: GeoJSON export, before/after window, decision trace, Find Similar,
the evaluation manifest, negative-evidence metadata and MGRS on results."""

import json
import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog.database import init_connection, init_schema
from change_detection import run_change_detection
from main import app
from mgrs_ref import to_mgrs
from s2_factory import Scenario, ingest_without_embedding, make_safe

NAME_A = "S2A_MSIL2A_20240110T050649_N0510_R019_T43PHM_20240110T090000"
NAME_B = "S2A_MSIL2A_20240214T050649_N0510_R019_T43PHM_20240214T090000"
CLEARED = (200, 260, 300, 380)  # vegetation -> bare, well inside the 512 x 512 scene
NEAR_EDGE = (200, 260, 20, 100)  # 20 px from the west edge
TILE_BOX = (40, 100, 40, 120)  # inside the first 224 px crop


def cleared_pair(tmp_path, box=CLEARED):
    """Two scenes, one vegetation->bare change, analysed without the (slow) embedding step."""
    a = ingest_without_embedding(
        make_safe(tmp_path / "src", NAME_A, Scenario(date="2024-01-10", noise_seed=1, landcover=[(*box, "vegetation")]))
    )
    b = ingest_without_embedding(
        make_safe(tmp_path / "src", NAME_B, Scenario(date="2024-02-14", noise_seed=2, landcover=[(*box, "bare")]))
    )
    run_change_detection(a, b)
    return TestClient(app)


def only_candidate(client):
    body = client.get("/api/changes").json()
    assert body["total"] == 1
    return body["candidates"][0]


# ---- GeoJSON export ----------------------------------------------------------------------------------------------


def test_export_is_geojson_with_the_provenance_record(tmp_path):
    client = cleared_pair(tmp_path)
    resp = client.get("/api/export/changes")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/geo+json")
    assert re.fullmatch(r'attachment; filename="iris_changes_\d{8}\.geojson"', resp.headers["content-disposition"])

    fc = resp.json()
    assert fc["type"] == "FeatureCollection"
    assert fc["crs"] == {"type": "name", "properties": {"name": "EPSG:4326"}}
    assert fc["feature_count"] == len(fc["features"]) == 1
    (feature,) = fc["features"]
    cand = only_candidate(client)

    p = feature["properties"]
    assert feature["type"] == "Feature" and feature["id"] == p["candidate_id"] == cand["candidate_id"]
    assert {
        "candidate_id", "direction", "change_type", "confidence", "confidence_breakdown", "mgrs", "scene_a_id",
        "scene_b_id", "date_a", "date_b", "sensor", "delta_ndvi", "analyst_decision", "analyst_id", "reviewed_at",
        "processing",
    } <= set(p)
    assert (p["direction"], p["change_type"]) == ("disappearance", "clearance")
    assert p["confidence"] == pytest.approx(cand["confidence"], abs=1e-4)
    assert set(p["confidence_breakdown"]) == {
        "alignment_quality", "cluster_distance_percentile", "terrain_flatness", "valid_coverage",
    }
    assert p["confidence_breakdown"]["alignment_quality"] == 0.9 and p["confidence_breakdown"]["terrain_flatness"] == 1.0
    assert p["mgrs"] == cand["mgrs"] and re.fullmatch(r"43[A-Z]{3}\d{10}", p["mgrs"])
    assert (p["scene_a_id"], p["scene_b_id"], p["date_a"], p["date_b"], p["sensor"]) == (
        NAME_A, NAME_B, "2024-01-10", "2024-02-14", "sentinel2",
    )
    assert p["delta_ndvi"] < -0.15
    assert (p["analyst_decision"], p["analyst_id"], p["reviewed_at"]) == ("pending", None, None)
    proc = p["processing"]
    assert proc["masking"] == "SCL" and proc["alignment"] == "same_mgrs_grid" and proc["normalisation"] == "pif_robust_regression"
    assert 0 < proc["cloud_trust_a"] <= 1 and 0 < proc["cloud_trust_b"] <= 1
    assert p["area_px"] == cand["area_px"] and p["area_ha"] == pytest.approx(cand["area_px"] / 100.0, abs=0.01)
    assert p["raw_checksum_a"] and p["raw_checksum_b"]


def test_export_geometry_is_the_outline_not_the_box(tmp_path):
    client = cleared_pair(tmp_path)
    feature = client.get("/api/export/changes").json()["features"][0]
    cand = only_candidate(client)
    assert feature["properties"]["geometry_source"] == "changed_pixel_outline"
    geom = feature["geometry"]
    assert geom["type"] in ("Polygon", "MultiPolygon")
    ring = geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"][0][0]
    assert ring[0] == ring[-1] and len(ring) >= 5
    lons, lats = [p[0] for p in ring], [p[1] for p in ring]
    min_lon, min_lat, max_lon, max_lat = cand["bounds"]
    assert min_lon - 1e-6 <= min(lons) and max(lons) <= max_lon + 1e-6
    assert min_lat - 1e-6 <= min(lats) and max(lats) <= max_lat + 1e-6


def test_export_carries_the_review_and_falls_back_to_the_box_for_old_candidates(tmp_path):
    client = cleared_pair(tmp_path)
    cid = only_candidate(client)["candidate_id"]
    assert client.post(f"/api/changes/{cid}/review", json={"decision": "confirmed", "notes": "site visit"}).status_code == 200
    p = client.get("/api/export/changes").json()["features"][0]["properties"]
    assert p["analyst_decision"] == "confirmed" and p["analyst_id"] and p["reviewed_at"]
    assert p["analyst_notes"] == "site visit" and p["review_count"] == 1

    conn = init_connection()
    try:
        conn.execute("UPDATE change_candidates SET geometry = NULL, mgrs_ref = NULL, centroid_lon = NULL, centroid_lat = NULL")
        conn.commit()
    finally:
        conn.close()
    feature = client.get("/api/export/changes").json()["features"][0]
    assert feature["properties"]["geometry_source"] == "bounding_box"
    assert feature["geometry"]["type"] == "Polygon" and len(feature["geometry"]["coordinates"][0]) == 5
    assert re.fullmatch(r"43[A-Z]{3}\d{10}", feature["properties"]["mgrs"])  # derived from the box centre instead


def test_export_can_be_limited_and_is_empty_without_candidates(tmp_path):
    client = cleared_pair(tmp_path)
    assert client.get("/api/export/changes", params={"min_confidence": 0.999}).json()["feature_count"] == 0
    assert client.get("/api/export/changes", params={"job_id": 999}).json()["features"] == []


# ---- candidate detail: window, MGRS, decision trace --------------------------------------------------------------


def test_detail_opens_on_a_window_four_times_the_box_inside_the_scene(tmp_path):
    client = cleared_pair(tmp_path)
    cand = only_candidate(client)
    d = client.get(f"/api/changes/{cand['candidate_id']}").json()
    min_lon, min_lat, max_lon, max_lat = d["bounds"]
    w = d["display_bounds"]
    assert (w[2] - w[0]) == pytest.approx(4 * (max_lon - min_lon), rel=1e-6)
    assert (w[3] - w[1]) == pytest.approx(4 * (max_lat - min_lat), rel=1e-6)
    assert w[0] < min_lon and w[2] > max_lon and w[1] < min_lat and w[3] > max_lat  # context on every side
    e = d["scene_b"]["bounds_wgs84"]
    assert e[0] <= w[0] and w[2] <= e[2] and e[1] <= w[1] and w[3] <= e[3]  # clamped to the scene
    assert d["mgrs"] == cand["mgrs"] and d["centroid"] and d["sub_blobs"] == 1


def test_detail_window_stays_inside_the_scene_near_its_edge(tmp_path):
    client = cleared_pair(tmp_path, NEAR_EDGE)
    d = client.get(f"/api/changes/{only_candidate(client)['candidate_id']}").json()
    min_lon, min_lat, max_lon, max_lat = d["bounds"]
    w, e = d["display_bounds"], d["scene_b"]["bounds_wgs84"]
    assert w[0] == pytest.approx(e[0], abs=1e-9)  # slid back to the scene edge...
    assert (w[2] - w[0]) == pytest.approx(4 * (max_lon - min_lon), rel=1e-6)  # ...without losing its size
    assert w[0] <= min_lon and w[2] >= max_lon


def test_detail_lists_the_decision_trace_in_pipeline_order(tmp_path):
    client = cleared_pair(tmp_path)
    d = client.get(f"/api/changes/{only_candidate(client)['candidate_id']}").json()
    lines = {line["key"]: line["text"] for line in d["processing_details"]}
    assert [line["key"] for line in d["processing_details"]] == ["masking", "alignment", "normalisation", "detection", "direction", "evidence"]
    assert re.fullmatch(r"SCL-based, CloudTrust A: \d\.\d\d, CloudTrust B: \d\.\d\d", lines["masking"])
    assert lines["alignment"] == "Same MGRS tile (T43PHM) — grid verified by definition"
    assert re.fullmatch(r"PIF robust regression, [\d,]+ anchor pixels, gain: \d\.\d\d, offset: -?\d+\.\d", lines["normalisation"])
    assert re.fullmatch(r"NIR difference → K-Means K=2, change centroid magnitude: \d+\.\d", lines["detection"])
    assert lines["direction"] == "Disappearance — dominant class A: vegetation, dominant class B: bare"
    assert re.fullmatch(r"ΔNDVI: -\d\.\d\d", lines["evidence"])


def test_a_candidate_without_direction_evidence_says_so(tmp_path):
    client = cleared_pair(tmp_path)
    cid = only_candidate(client)["candidate_id"]
    conn = init_connection()
    try:
        conn.execute("UPDATE change_candidates SET direction = NULL, direction_evidence = NULL, change_type = 'vegetation_loss'")
        conn.commit()
    finally:
        conn.close()
    lines = {line["key"]: line["text"] for line in client.get(f"/api/changes/{cid}").json()["processing_details"]}
    assert lines["direction"].startswith("Not classified")


def test_browse_list_reports_the_analysed_area_for_negative_evidence(tmp_path):
    client = cleared_pair(tmp_path)
    area = client.get("/api/changes").json()["area"]
    assert len(area["bounds"]) == 4 and area["bounds"][0] < area["bounds"][2]
    assert area["mgrs"] == to_mgrs((area["bounds"][1] + area["bounds"][3]) / 2, (area["bounds"][0] + area["bounds"][2]) / 2)


# ---- with embeddings: Find Similar, evaluation manifest, negative evidence, MGRS on results ------------------------


def ingest_via_api(client, tmp_path, name, scn):
    resp = client.post("/api/ingest", json={"file_path": str(make_safe(tmp_path / "src", name, scn))})
    assert resp.status_code == 200, resp.text


@pytest.fixture
def archive(tmp_path):
    """Both scenes imported through the API (embedded and indexed); one clear change inside the first crop."""
    client = TestClient(app)
    ingest_via_api(client, tmp_path, NAME_A, Scenario(date="2024-01-10", noise_seed=1, landcover=[(*TILE_BOX, "vegetation")]))
    ingest_via_api(client, tmp_path, NAME_B, Scenario(date="2024-02-14", noise_seed=2, landcover=[(*TILE_BOX, "bare")]))
    return client


def tiles_of(scene_id):
    conn = init_connection()
    try:
        return {r[0]: r[1] for r in conn.execute("SELECT tile_id, mgrs_ref FROM tiles WHERE scene_id = ?", (scene_id,))}
    finally:
        conn.close()


def test_find_similar_uses_the_seeds_own_embedding(archive):
    tiles_b = tiles_of(NAME_B)
    assert len(tiles_b) == 4 and all(re.fullmatch(r"43[A-Z]{3}\d{10}", m) for m in tiles_b.values())  # stored at ingestion
    seed = next(t for t in tiles_b if t.endswith("crop_00224_00224"))  # a tile the change did not touch

    resp = archive.post("/api/search/similar", json={"tile_id": seed})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["label"] == f"Similar to {seed}"
    assert body["similar_to"]["tile_id"] == seed and body["similar_to"]["scene_id"] == NAME_B
    assert body["similar_to"]["mgrs"] == tiles_b[seed]

    results = body["semantic_results"]
    assert len(results) == 7  # the whole archive (8 tiles) except the seed itself
    assert seed not in [r["tile_id"] for r in results]
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True) and all(s <= 1.0 for s in scores)
    assert {r["scene_id"] for r in results} == {NAME_A, NAME_B}  # across dates: one find leads to many
    twin = results[0]  # the same ground on the other date looks almost the same
    assert twin["scene_id"] == NAME_A and twin["tile_id"].endswith("crop_00224_00224") and twin["score"] > 0.9
    assert all(re.fullmatch(r"43[A-Z]{3}\d{10}", r["mgrs"]) for r in results)

    assert len(archive.post("/api/search/similar", json={"tile_id": seed, "top_k": 3}).json()["semantic_results"]) == 3


def test_find_similar_from_a_change_candidate_seeds_with_its_after_tile(archive):
    cand = archive.get("/api/changes").json()["candidates"][0]
    resp = archive.post("/api/search/similar", json={"candidate_id": cand["candidate_id"]})
    assert resp.status_code == 200, resp.text
    seed = resp.json()["similar_to"]
    assert seed["scene_id"] == NAME_B and seed["tile_id"].endswith("crop_00000_00000")
    assert seed["candidate_id"] == cand["candidate_id"]
    assert len(resp.json()["semantic_results"]) == 7


def test_find_similar_rejects_bad_seeds(archive):
    assert archive.post("/api/search/similar", json={}).status_code == 422
    assert archive.post("/api/search/similar", json={"tile_id": "a_b", "candidate_id": 1}).status_code == 422
    assert archive.post("/api/search/similar", json={"tile_id": "no_such_tile"}).status_code == 404
    assert archive.post("/api/search/similar", json={"candidate_id": 424242}).status_code == 404
    assert archive.post("/api/search/similar", json={"tile_id": "../../etc/passwd"}).status_code == 422


def test_search_results_carry_mgrs(archive):
    body = archive.post("/api/search", json={"query": "cleared land", "top_k": 5}).json()
    assert body["semantic_results"] and all(re.fullmatch(r"43[A-Z]{3}\d{10}", r["mgrs"]) for r in body["semantic_results"])
    for r in body["change_results"]:
        assert re.fullmatch(r"43[A-Z]{3}\d{10}", r["mgrs"]) and r["sub_blobs"] >= 1


def test_the_evaluation_manifest_records_stages_storage_index_latency_and_hardware(archive):
    archive.post("/api/search", json={"query": "cleared land", "top_k": 5})
    tile = next(iter(tiles_of(NAME_B)))
    archive.post("/api/search/similar", json={"tile_id": tile})

    resp = archive.get("/api/eval/manifest")
    assert resp.status_code == 200
    m = resp.json()
    assert m["schema"] == "iris.eval_manifest/1" and m["generated_at"]

    # per-stage wall-clock time of each import
    assert len(m["ingestions"]) == 2
    for ing in m["ingestions"]:
        assert {"format_detection", "crop_generation", "embedding", "faiss_insertion"} <= set(ing["stages_s"]), ing["stages_s"]
        assert all(v >= 0 for v in ing["stages_s"].values()) and ing["total_s"] > 0
        assert ing["tiles_indexed"] == 4 and ing["outcome"] == "completed"

    # per-stage time of change detection (pipeline phases under the report's names)
    (job,) = m["change_detection"]
    assert {"masking", "grid_check", "alignment", "normalisation", "detection", "classification", "scoring"} <= set(job["stages_s"])
    assert "radiometry" not in job["stages_s"] and job["status"] == "completed" and job["detections"] == 1

    s = m["storage"]
    assert s["cogs_bytes"] > 0 and s["crops_bytes"] > 0 and s["faiss_index_bytes"] > 0 and s["sqlite_bytes"] > 0
    assert s["total_bytes"] == s["cogs_bytes"] + s["crops_bytes"] + s["faiss_index_bytes"] + s["sqlite_bytes"]
    assert (m["index"]["scenes"], m["index"]["tiles"], m["index"]["faiss_vectors"]) == (2, 8, 8)
    assert m["index"]["change_candidates"] == 1

    q = m["queries"]
    assert q["text_search"]["count"] == 1 and q["find_similar"]["count"] == 1
    for kind in q.values():
        assert kind["p50_ms"] <= kind["p95_ms"] <= kind["p99_ms"] <= kind["max_ms"] and kind["mean_ms"] > 0

    hw = m["hardware"]
    assert hw["cpu_model"] and hw["cpu_threads"] >= 1 and hw["ram_total_gb"] > 0 and "gpu" in hw and hw["os"]
    assert m["software"]["mgrs"]

    # the same document is kept on disk, updated after every operation
    on_disk = json.loads(Path("data/eval_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["index"] == m["index"] and len(on_disk["ingestions"]) == 2


def test_the_manifest_is_updated_by_each_operation_without_being_asked(tmp_path):
    client = TestClient(app)
    ingest_via_api(client, tmp_path, NAME_A, Scenario(date="2024-01-10", noise_seed=1))
    on_disk = json.loads(Path("data/eval_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["last_operation"].startswith("ingestion:") and len(on_disk["ingestions"]) == 1
    assert on_disk["index"]["scenes"] == 1

    client.post("/api/search", json={"query": "forest", "top_k": 3})
    on_disk = json.loads(Path("data/eval_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["last_operation"] == "query:text_search" and on_disk["queries"]["text_search"]["count"] == 1

    assert client.delete(f"/api/scenes/{NAME_A}").status_code == 200
    on_disk = json.loads(Path("data/eval_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["last_operation"].startswith("scene_deleted:") and on_disk["index"]["scenes"] == 0


def test_negative_evidence_metadata_names_the_area_dates_and_coverage(tmp_path):
    client = TestClient(app)
    ingest_via_api(client, tmp_path, NAME_A, Scenario(date="2024-01-10", noise_seed=1))
    ingest_via_api(client, tmp_path, NAME_B, Scenario(date="2024-02-14", noise_seed=2))  # nothing changed

    body = client.post("/api/search", json={"query": "new structures near river", "top_k": 5}).json()
    assert body["change_status"] == "no_change_detected"
    meta = body["change_meta"]
    assert meta["dates_analysed"] == ["2024-01-10", "2024-02-14"]
    assert 0.5 < meta["valid_coverage"] <= 1.0
    bounds = meta["area_bounds"]
    assert len(bounds) == 4 and bounds[0] < bounds[2] and bounds[1] < bounds[3]
    assert 72 < bounds[0] < 74 and 27 < bounds[1] < 29
    assert meta["mgrs"] == to_mgrs((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2)
