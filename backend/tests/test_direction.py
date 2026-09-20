"""Tests for Phase 4b: per-date classification, the direction rules, refined change types, and the search boost."""

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog import changes
from catalog.database import init_connection, init_schema, list_scene_tiles
from change_detection import params, run_change_detection
from change_detection.direction import (
    BARE,
    OTHER,
    VEGETATION,
    WATER,
    classify_strip,
    decide,
    direction_hints,
)
from embedding.embedder import RemoteCLIPEmbedder
from embedding.index import get_vector_store
from main import app
from s2_factory import Scenario, ingest_without_embedding, make_safe

NAME_A = "S2A_MSIL2A_20240110T050649_N0510_R019_T43PHM_20240110T090000"
NAME_B = "S2A_MSIL2A_20240214T050649_N0510_R019_T43PHM_20240214T090000"
BOX = (200, 260, 300, 380)  # 60 x 80 = 4800 px
TILE_BOX = (40, 100, 40, 120)  # inside the first 224 px crop, so exactly one tile covers it


def counts(**kw) -> np.ndarray:
    """Class counts in the order OTHER, VEGETATION, WATER, BARE."""
    out = np.zeros(4, dtype=np.int64)
    for name, code in (("other", OTHER), ("veg", VEGETATION), ("water", WATER), ("bare", BARE)):
        out[code] = kw.get(name, 0)
    return out


# ---- Step 1: per-date classification ----------------------------------------------------------------------------


def test_classification_thresholds():
    # reflectance columns: B02, B03, B04, B08
    px = np.array(
        [
            [0.03, 0.06, 0.04, 0.40],  # NDVI 0.82                    -> vegetation
            [0.10, 0.13, 0.17, 0.22],  # NDVI 0.13, NDWI < 0           -> bare
            [0.06, 0.05, 0.03, 0.01],  # NDWI 0.67                    -> water (also NDVI < 0.15, water wins)
            [0.05, 0.08, 0.10, 0.15],  # NDVI 0.20: between the limits -> other
        ],
        dtype=np.float32,
    )
    refl = px.T.reshape(4, 1, 4)
    assert classify_strip(refl)[0].tolist() == [VEGETATION, BARE, WATER, OTHER]


def test_classification_of_nodata_is_not_vegetation_or_water():
    refl = np.zeros((4, 1, 2), dtype=np.float32)  # sum is 0: NDVI and NDWI are undefined
    assert classify_strip(refl)[0].tolist() == [OTHER, OTHER]


# ---- Step 2/3: the rules -----------------------------------------------------------------------------------------


def test_vegetation_to_bare_is_disappearance_and_clearance():
    d, t, ev = decide(counts(veg=900, bare=100), counts(bare=950, other=50), -0.31, -0.2)
    assert (d, t) == ("disappearance", "clearance")
    assert ev["dominant_a"] == "vegetation" and ev["dominant_b"] == "bare" and ev["delta_ndvi"] == -0.31
    assert ev["swir_available"] is False


def test_water_to_bare_is_disappearance_and_water_loss():
    d, t, _ = decide(counts(water=1000), counts(bare=900, water=100), 0.1, 0.0)
    assert (d, t) == ("disappearance", "water_loss")


def test_bare_to_vegetation_is_appearance_and_revegetation():
    d, t, _ = decide(counts(bare=1000), counts(veg=800, bare=200), 0.5, -0.3)
    assert (d, t) == ("appearance", "revegetation")


def test_bare_to_brighter_bare_with_falling_ndvi_is_construction():
    d, t, ev = decide(counts(bare=1000), counts(bare=1000), -0.09, 0.4)
    assert (d, t) == ("appearance", "construction")
    assert "SWIR-free" in ev["rule"]


def test_bare_to_bare_without_a_real_ndvi_drop_is_not_construction():
    # The bare threshold is NDVI < 0.15, so bare -> bare can never fall by 0.15; the separate, smaller
    # NDVI_DIRECTION_MIN is what counts as a fall. Below it (or without brightening), nothing is claimed.
    assert decide(counts(bare=1000), counts(bare=1000), -0.02, 0.4)[:2] == ("unclassified", "unclassified")
    assert decide(counts(bare=1000), counts(bare=1000), -0.2, 0.05)[:2] == ("unclassified", "unclassified")


def test_same_dominant_class_growing_or_shrinking_by_area():
    # 1.2x and 0.8x are the documented limits, both strict
    grew = decide(counts(bare=500, other=500), counts(bare=700, other=300), None, None)
    assert grew[0] == "expansion" and grew[1] == "urban_expansion"
    shrank = decide(counts(veg=800, other=200), counts(veg=500, other=500), None, None)
    assert shrank[0] == "contraction" and shrank[1] == "vegetation_retreat"
    edge = decide(counts(veg=500, other=500), counts(veg=600, other=400), None, None)  # exactly 1.2x
    assert edge[0] == "unclassified"


def test_no_rule_means_unclassified_not_a_guess():
    d, t, ev = decide(counts(veg=1000), counts(veg=1000), 0.0, 0.0)
    assert (d, t) == ("unclassified", "unclassified") and ev["rule"] == "no direction rule matched"
    # vegetation to water is in none of the rules
    assert decide(counts(veg=1000), counts(water=1000), -0.5, 0.0)[:2] == ("unclassified", "unclassified")


def test_direction_without_ndvi_still_reports_a_direction():
    d, t, ev = decide(counts(veg=1000), counts(bare=1000), None, None)
    assert d == "disappearance" and t == "unclassified" and ev["delta_ndvi"] is None  # type needs the NDVI drop


# ---- Part C: keywords ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query, expected",
    [
        ("new construction near the river", ["appearance"]),
        ("forest cleared for farmland", ["disappearance"]),
        ("expanding settlement, sprawl", ["expansion"]),
        ("lake shrinking", ["contraction"]),
        ("new road and demolished warehouses", ["appearance", "disappearance"]),
        ("river near a bridge", []),
        ("urban renewal", []),  # whole words only: "renewal" is not "new"
        ("NEW Buildings", ["appearance"]),
    ],
)
def test_direction_hints(query, expected):
    assert direction_hints(query) == expected


# ---- end to end: land-cover transitions through the real pipeline -------------------------------------------------


def run(tmp_path, kind_a: str, kind_b: str):
    a = ingest_without_embedding(
        make_safe(tmp_path / "src", NAME_A, Scenario(date="2024-01-10", noise_seed=1, landcover=[(*BOX, kind_a)]))
    )
    b = ingest_without_embedding(
        make_safe(tmp_path / "src", NAME_B, Scenario(date="2024-02-14", noise_seed=2, landcover=[(*BOX, kind_b)]))
    )
    run_change_detection(a, b)
    conn = init_connection()
    try:
        init_schema(conn)
        return changes.list_candidates(conn), changes.get_job_for_pair(conn, a, b)
    finally:
        conn.close()


@pytest.mark.parametrize(
    "kind_a, kind_b, direction, change_type, dominant",
    [
        ("vegetation", "bare", "disappearance", "clearance", ("vegetation", "bare")),
        ("water", "bare", "disappearance", "water_loss", ("water", "bare")),
        ("bare", "vegetation", "appearance", "revegetation", ("bare", "vegetation")),
        ("bare", "built", "appearance", "construction", ("bare", "bare")),
    ],
)
def test_pipeline_names_the_transition(tmp_path, kind_a, kind_b, direction, change_type, dominant):
    cands, job = run(tmp_path, kind_a, kind_b)
    assert len(cands) == 1
    cand = cands[0]
    assert (cand["direction"], cand["change_type"]) == (direction, change_type)
    ev = cand["direction_evidence"]
    assert (ev["dominant_a"], ev["dominant_b"]) == dominant
    assert ev["share_a"][dominant[0]] > 0.9 and ev["swir_available"] is False
    assert job["details"]["direction"]["by_direction"] == {direction: 1}
    assert job["details"]["timings_s"]["direction"] >= 0


def test_direction_is_stored_with_the_candidate_and_shown_by_the_api(tmp_path):
    run(tmp_path, "vegetation", "bare")
    client = TestClient(app)
    body = client.get("/api/changes").json()
    cand = body["candidates"][0]
    assert cand["direction"] == "disappearance" and cand["change_type"] == "clearance"

    detail = client.get(f"/api/changes/{cand['candidate_id']}").json()
    assert detail["direction_evidence"]["dominant_a"] == "vegetation"
    assert detail["direction_evidence"]["delta_ndvi"] == pytest.approx(cand["mean_dndvi"], abs=1e-3)

    assert client.get("/api/changes", params={"directions": "disappearance"}).json()["matching"] == 1
    assert client.get("/api/changes", params={"directions": "appearance"}).json()["matching"] == 0
    assert client.get("/api/changes", params={"types": "clearance"}).json()["matching"] == 1
    assert client.get("/api/changes", params={"types": "clearance", "directions": "appearance"}).json()["matching"] == 0


def test_legacy_rows_are_presented_as_unclassified(tmp_path):
    """A candidate stored before Phase 4b has the old type name and no direction; it must not break filters."""
    run(tmp_path, "vegetation", "bare")
    conn = init_connection()
    try:
        conn.execute("UPDATE change_candidates SET change_type = 'vegetation_loss', direction = NULL, direction_evidence = NULL")
        conn.commit()
    finally:
        conn.close()
    client = TestClient(app)
    cand = client.get("/api/changes").json()["candidates"][0]
    assert cand["change_type"] == "unclassified" and cand["direction"] == "unclassified"
    assert client.get("/api/changes", params={"types": "unclassified"}).json()["matching"] == 1
    assert client.get("/api/changes", params={"directions": "unclassified"}).json()["matching"] == 1
    assert client.get("/api/changes", params={"types": "clearance"}).json()["matching"] == 0
    assert client.get(f"/api/changes/{cand['candidate_id']}").json()["direction_evidence"] is None


# ---- search: the direction boost ----------------------------------------------------------------------------------


def test_search_boosts_matching_direction_by_the_configured_amount(tmp_path, monkeypatch):
    client = TestClient(app)
    for name, scn in (
        (NAME_A, Scenario(date="2024-01-10", noise_seed=1, landcover=[(*TILE_BOX, "vegetation")])),
        (NAME_B, Scenario(date="2024-02-14", noise_seed=2, landcover=[(*TILE_BOX, "bare")])),
    ):
        assert client.post("/api/ingest", json={"file_path": str(make_safe(tmp_path / "src", name, scn))}).status_code == 200
    conn = init_connection()
    try:
        tile = next(t for t in list_scene_tiles(conn, NAME_B) if t["tile_id"].endswith("crop_00000_00000"))
    finally:
        conn.close()
    vec = get_vector_store().vectors_for([tile["faiss_id"]])[0].reshape(1, -1)  # what the change's tile looks like
    monkeypatch.setattr(RemoteCLIPEmbedder, "embed_text", lambda self, text: vec)

    def top(query):
        resp = client.post("/api/search", json={"query": query, "top_k": 5})
        assert resp.status_code == 200, resp.text
        return resp.json()

    plain = top("river bank")
    assert plain["change_meta"]["direction_hints"] == []
    assert all(r["direction_boost"] == 0 for r in plain["change_results"])

    cleared = top("river bank cleared")  # "cleared" -> disappearance
    assert cleared["change_meta"]["direction_hints"] == ["disappearance"]
    hit = cleared["change_results"][0]
    assert hit["direction"] == "disappearance" and hit["direction_boost"] == params.DIRECTION_SEARCH_BOOST == 0.2
    base = plain["change_results"][0]["combined_score"]
    assert hit["combined_score"] == pytest.approx(base + 0.2, abs=1e-3)
    assert hit["evidence"]["dominant_a"] == "vegetation"

    built = top("new river bank")  # "new" -> appearance: this candidate is a disappearance, so no boost
    assert built["change_meta"]["direction_hints"] == ["appearance"]
    assert built["change_results"][0]["direction_boost"] == 0
