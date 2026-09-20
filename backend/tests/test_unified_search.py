"""Tests for the unified search: one query -> semantic_results + change_results + change_status."""

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog.database import init_connection, list_scene_tiles
from change_detection import params
from change_detection.direction import CHANGE_TYPES, DIRECTIONS
from embedding.embedder import RemoteCLIPEmbedder
from embedding.index import get_vector_store
from main import app
from s2_factory import Scenario, make_safe

NAME_A = "S2A_MSIL2A_20240110T050649_N0510_R019_T43PHM_20240110T090000"
NAME_B = "S2A_MSIL2A_20240214T050649_N0510_R019_T43PHM_20240214T090000"
# Sits entirely inside the first 224 px crop of the scene, so exactly one tile covers it
PATCH = (40, 100, 40, 120, 0.25)
PATCH_TILE = "crop_00000_00000"


def ingest(client, tmp_path, name, scn):
    resp = client.post("/api/ingest", json={"file_path": str(make_safe(tmp_path / "src", name, scn))})
    assert resp.status_code == 200, resp.text


@pytest.fixture
def archive(tmp_path):
    """A (before) and B (after, with one clear change in the first tile)."""
    client = TestClient(app)
    ingest(client, tmp_path, NAME_A, Scenario(date="2024-01-10", noise_seed=1))
    ingest(client, tmp_path, NAME_B, Scenario(date="2024-02-14", noise_seed=2, patches=[PATCH]))
    return client


def tile_vector(scene_id, tile_suffix):
    conn = init_connection()
    try:
        tile = next(t for t in list_scene_tiles(conn, scene_id) if t["tile_id"].endswith(tile_suffix))
    finally:
        conn.close()
    return get_vector_store().vectors_for([tile["faiss_id"]])[0]


def query_as(monkeypatch, vector):
    """Make the text encoder return this vector, so a chosen tile is (or is not) what the query 'means'."""
    v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    monkeypatch.setattr(RemoteCLIPEmbedder, "embed_text", lambda self, text: v)


def search(client, query="new structures near river", **kw):
    resp = client.post("/api/search", json={"query": query, "top_k": 10, **kw})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_response_has_both_sections_and_the_documented_shape(archive):
    body = search(archive, scene_id=NAME_B)
    assert set(body) >= {"semantic_results", "change_results", "change_status", "change_meta"}
    assert set(body["change_meta"]) >= {"dates_compared", "valid_coverage", "total_candidates", "matching_candidates"}
    assert body["change_status"] in ("found", "no_match")


def test_semantic_results_are_scoped_to_the_selected_scene(archive):
    for scene in (NAME_A, NAME_B):
        body = search(archive, scene_id=scene)
        assert len(body["semantic_results"]) == 4  # every tile of that scene, best first
        assert {r["scene_id"] for r in body["semantic_results"]} == {scene}
        scores = [r["score"] for r in body["semantic_results"]]
        assert scores == sorted(scores, reverse=True)

    everywhere = search(archive)  # no scene: the whole index
    assert {r["scene_id"] for r in everywhere["semantic_results"]} == {NAME_A, NAME_B}


def test_scoped_search_is_exact_cosine_over_the_scenes_tiles(archive, monkeypatch):
    v = tile_vector(NAME_B, PATCH_TILE)
    query_as(monkeypatch, v)
    top = search(archive, scene_id=NAME_B)["semantic_results"]
    assert top[0]["tile_id"].endswith(PATCH_TILE) and top[0]["score"] == pytest.approx(1.0, abs=1e-4)


def test_matching_change_is_found_and_ranked_with_its_evidence(archive, monkeypatch):
    query_as(monkeypatch, tile_vector(NAME_B, PATCH_TILE))  # the after-crop over the change IS what the query means
    body = search(archive, scene_id=NAME_B)

    assert body["change_status"] == "found"
    assert body["change_meta"]["matching_candidates"] == len(body["change_results"]) >= 1
    assert body["change_meta"]["total_candidates"] >= body["change_meta"]["matching_candidates"]
    assert body["change_meta"]["dates_compared"] == "2024-01-10 → 2024-02-14"
    assert 0.5 < body["change_meta"]["valid_coverage"] <= 1.0

    hit = body["change_results"][0]
    assert hit["matches_query"] is True and hit["semantic_match_score"] == 1.0
    assert hit["semantic_similarity"] == pytest.approx(1.0, abs=1e-3)
    assert hit["after_tile_id"].endswith(PATCH_TILE)
    assert hit["combined_score"] == pytest.approx(0.5 * hit["semantic_match_score"] + 0.5 * hit["confidence"], abs=1e-4)
    # everything the analyst is shown for a change
    assert hit["change_type"] in CHANGE_TYPES and hit["direction"] in DIRECTIONS
    assert hit["scene_a_date"] == "2024-01-10" and hit["scene_b_date"] == "2024-02-14"
    assert hit["mean_dndvi"] is not None and 0 < hit["confidence"] <= 1 and len(hit["bounds"]) == 4
    assert "direction" in hit and hit["review_status"] == "pending"


def test_change_results_are_ordered_by_combined_score(tmp_path, monkeypatch):
    client = TestClient(app)
    ingest(client, tmp_path, NAME_A, Scenario(date="2024-01-10", noise_seed=1))
    ingest(client, tmp_path, NAME_B, Scenario(date="2024-02-14", noise_seed=2, patches=[PATCH, (40, 100, 260, 340, 0.6)]))
    query_as(monkeypatch, tile_vector(NAME_B, PATCH_TILE))
    body = search(client, scene_id=NAME_B)

    combined = [r["combined_score"] for r in body["change_results"]]
    assert len(combined) >= 1 and combined == sorted(combined, reverse=True)
    assert all(r["matches_query"] for r in body["change_results"])


def test_changes_exist_but_none_match_the_query(archive, monkeypatch):
    query_as(monkeypatch, -tile_vector(NAME_B, PATCH_TILE))  # the opposite of what the after-crop looks like
    body = search(archive, scene_id=NAME_B)

    assert body["change_status"] == "no_match"
    assert body["change_results"] == []
    assert body["change_meta"]["total_candidates"] >= 1 and body["change_meta"]["matching_candidates"] == 0
    assert body["semantic_results"]  # the semantic section is unaffected


def test_analysed_area_with_no_change_is_a_confirmed_negative(tmp_path):
    client = TestClient(app)
    ingest(client, tmp_path, NAME_A, Scenario(date="2024-01-10", noise_seed=1))
    ingest(client, tmp_path, NAME_B, Scenario(date="2024-02-14", noise_seed=2))  # nothing changed
    body = search(client, scene_id=NAME_B)

    assert body["change_status"] == "no_change_detected"
    assert body["change_results"] == []
    meta = body["change_meta"]
    assert meta["dates_compared"] == "2024-01-10 → 2024-02-14" and meta["total_candidates"] == 0
    assert 0.9 < meta["valid_coverage"] <= 1.0  # the evidence behind the negative: how much of the area was observed
    assert meta["pairs"] and meta["pairs"][0]["candidates"] == 0


def test_single_scene_has_no_comparison(tmp_path):
    client = TestClient(app)
    ingest(client, tmp_path, NAME_A, Scenario(date="2024-01-10", noise_seed=1))
    body = search(client, scene_id=NAME_A)

    assert body["change_status"] == "no_comparison_available"
    assert body["change_results"] == [] and body["change_meta"]["total_candidates"] == 0
    assert body["semantic_results"]


def test_a_pair_that_could_not_be_analysed_is_not_reported_as_no_change(tmp_path):
    client = TestClient(app)
    ingest(client, tmp_path, NAME_A, Scenario(date="2024-01-10", noise_seed=1))
    ingest(client, tmp_path, NAME_B, Scenario(date="2024-02-14", noise_seed=2, nodata_rows=400))  # too little valid data
    body = search(client, scene_id=NAME_B)

    assert body["change_status"] == "no_comparison_available"  # "not analysed" is not "analysed, nothing there"
    assert "insufficient evidence" in body["change_meta"]["note"]


def test_scoping_validation(archive):
    assert archive.post("/api/search", json={"query": "forest", "scene_id": "no_such_scene"}).status_code == 404
    assert archive.post("/api/search", json={"query": "forest", "scene_id": "bad id!"}).status_code == 422
    assert archive.post("/api/search", json={"query": "  "}).status_code == 400


def test_search_from_the_older_scene_still_finds_the_change(archive, monkeypatch):
    """The scope is the AOI, not one date: from scene A the changes between A and B are still relevant."""
    query_as(monkeypatch, tile_vector(NAME_B, PATCH_TILE))
    body = search(archive, scene_id=NAME_A)
    assert {r["scene_id"] for r in body["semantic_results"]} == {NAME_A}
    assert body["change_status"] == "found" and body["change_results"][0]["scene_b_id"] == NAME_B


def test_negative_evidence_carries_what_is_needed_to_export_it(tmp_path):
    client = TestClient(app)
    ingest(client, tmp_path, NAME_A, Scenario(date="2024-01-10", noise_seed=1))
    ingest(client, tmp_path, NAME_B, Scenario(date="2024-02-14", noise_seed=2))
    meta = search(client, scene_id=NAME_B)["change_meta"]

    assert meta["thresholds"] == {
        "minimum_mapping_unit_px": params.MIN_BLOB_PIXELS,
        "minimum_confidence": params.MIN_STORED_CONFIDENCE,
    }
    assert meta["thresholds"]["minimum_mapping_unit_px"] == 100
    assert (meta["pairs"][0]["scene_a_id"], meta["pairs"][0]["scene_b_id"]) == (NAME_A, NAME_B)
    listing = client.get("/api/changes").json()  # the same evidence is available without running a search
    assert listing["thresholds"] == meta["thresholds"] and listing["pairs"][0]["valid_coverage"] == meta["pairs"][0]["valid_coverage"]
