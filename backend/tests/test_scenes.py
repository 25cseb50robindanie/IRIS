"""Tests for scene management: GET /api/scenes and DELETE /api/scenes/{scene_id}."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import api.ingest as ingest_module
from catalog.database import init_connection
from main import app
from s2_factory import Scenario, make_safe

NAME_A = "S2A_MSIL2A_20240110T050649_N0510_R019_T43PHM_20240110T090000"
NAME_B = "S2A_MSIL2A_20240214T050649_N0510_R019_T43PHM_20240214T090000"
PATCH = (200, 260, 300, 380, 0.35)


@pytest.fixture
def two_scenes(tmp_path):
    """Two overlapping scenes imported through the API, one change candidate, one analyst review."""
    client = TestClient(app)
    a = make_safe(tmp_path / "src", NAME_A, Scenario(date="2024-01-10", noise_seed=1))
    b = make_safe(tmp_path / "src", NAME_B, Scenario(date="2024-02-14", noise_seed=2, patches=[PATCH]))
    assert client.post("/api/ingest", json={"file_path": str(a)}).status_code == 200
    assert client.post("/api/ingest", json={"file_path": str(b)}).status_code == 200
    cid = client.get("/api/changes").json()["candidates"][0]["candidate_id"]
    assert client.post(f"/api/changes/{cid}/review", json={"decision": "confirmed"}).status_code == 200
    return client, a, b


def search(client, top_k=20):
    resp = client.post("/api/search", json={"query": "dense forest and vegetation", "top_k": top_k})
    assert resp.status_code == 200, resp.text
    return {r["tile_id"]: r for r in resp.json()["semantic_results"]}


def test_list_scenes_most_recent_acquisition_first(two_scenes):
    client, _, _ = two_scenes
    scenes = client.get("/api/scenes").json()["scenes"]

    assert [s["scene_id"] for s in scenes] == [NAME_B, NAME_A]
    newest = scenes[0]
    assert newest["acquisition_date"] == "2024-02-14" and newest["sensor"] == "sentinel2"
    assert newest["cog_available"] is True and newest["cog_url"].startswith("file:///")
    assert newest["tiles_count"] == 4 and newest["has_analysis"] is True
    assert len(newest["bounds_wgs84"]) == 4
    assert (newest["change_candidates"], newest["reviews"]) == (1, 1)
    assert Path(newest["cog_url"][len("file:///"):]).is_file()


def test_list_scenes_when_empty():
    assert TestClient(app).get("/api/scenes").json() == {"scenes": []}


def test_delete_removes_everything_the_scene_owns(two_scenes):
    client, a, b = two_scenes
    b_files = [Path(f"data/cogs/{NAME_B}{suffix}.tif") for suffix in ("", "_analysis", "_scl")]
    b_crops = Path(f"data/crops/{NAME_B}")
    assert all(f.is_file() for f in b_files) and b_crops.is_dir()
    a_before = {k: v for k, v in search(client).items() if v["scene_id"] == NAME_A}

    resp = client.delete(f"/api/scenes/{NAME_B}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["tiles"], body["change_candidates"], body["reviews"], body["jobs"]) == (4, 1, 1, 1)
    assert (body["vectors_removed"], body["vectors_remaining"]) == (4, 4)
    assert body["files_removed"] == 4 and body["files_failed"] == []  # display, analysis, SCL and the crop folder

    # gone from disk, the catalog and the index
    assert not any(f.exists() for f in b_files) and not b_crops.exists()
    assert [s["scene_id"] for s in client.get("/api/scenes").json()["scenes"]] == [NAME_A]
    assert client.get("/api/changes").json()["candidates"] == []
    assert client.get("/api/changes/jobs").json() == {"jobs": []}
    assert client.get(f"/api/status/{NAME_B}").status_code == 404
    assert client.get("/api/catalog/status").json()["faiss_vectors"] == 4
    assert {v["scene_id"] for v in search(client).values()} == {NAME_A}

    # the untouched scene keeps working exactly as before
    a_after = search(client)
    assert set(a_after) == set(a_before)
    for tile_id, hit in a_after.items():
        assert hit["score"] == pytest.approx(a_before[tile_id]["score"], abs=1e-4)
        assert hit["bounds"] == a_before[tile_id]["bounds"]

    # the analyst's original source files are never touched
    assert a.is_dir() and b.is_dir()


def test_deleting_the_older_scene_renumbers_the_newer_ones_vectors_correctly(two_scenes):
    """A's vectors are ids 0-3 and B's are 4-7; after deleting A, B must move to 0-3 without mixing anything up."""
    client, _, _ = two_scenes
    before = {k: v for k, v in search(client).items() if v["scene_id"] == NAME_B}
    assert len(before) == 4

    assert client.delete(f"/api/scenes/{NAME_A}").status_code == 200

    conn = init_connection()
    ids = sorted(r[0] for r in conn.execute("SELECT faiss_id FROM tiles").fetchall())
    conn.close()
    assert ids == [0, 1, 2, 3]

    after = search(client)
    assert set(after) == set(before)
    for tile_id, hit in after.items():  # same tile, same score, same location: the remap did not shuffle anything
        assert hit["scene_id"] == NAME_B
        assert hit["score"] == pytest.approx(before[tile_id]["score"], abs=1e-4)
        assert hit["bounds"] == before[tile_id]["bounds"]
    assert client.get("/api/changes").json()["total"] == 0  # the pair no longer exists


def test_deleting_the_last_scene_leaves_a_clean_empty_state(two_scenes):
    client, _, _ = two_scenes
    client.delete(f"/api/scenes/{NAME_B}")
    client.delete(f"/api/scenes/{NAME_A}")

    status = client.get("/api/catalog/status").json()
    assert (status["has_scenes"], status["faiss_vectors"], status["latest_scene"]) == (False, 0, None)
    assert client.get("/api/scenes").json() == {"scenes": []}
    assert client.post("/api/search", json={"query": "forest", "top_k": 3}).status_code == 409
    assert list(Path("data/cogs").glob("*.tif")) == []


def test_delete_unknown_or_malformed_scene_id(two_scenes):
    client, _, _ = two_scenes
    assert client.delete("/api/scenes/no_such_scene").status_code == 404
    assert client.delete("/api/scenes/bad id!").status_code == 400
    assert len(client.get("/api/scenes").json()["scenes"]) == 2


def test_delete_is_refused_while_an_import_is_processing(two_scenes):
    client, _, _ = two_scenes
    ingest_module.EMBED_LOCK.acquire()  # what a running embedding / change-detection job holds
    try:
        resp = client.delete(f"/api/scenes/{NAME_B}")
        assert resp.status_code == 409 and "still processing" in resp.json()["detail"]
    finally:
        ingest_module.EMBED_LOCK.release()
    assert len(client.get("/api/scenes").json()["scenes"]) == 2  # nothing was removed
    assert client.delete(f"/api/scenes/{NAME_B}").status_code == 200  # and it works once the job is done


def test_delete_never_removes_files_outside_the_app_data_folders(two_scenes, tmp_path):
    """A tampered catalog path must not turn delete into 'remove any file on disk'."""
    client, _, _ = two_scenes
    precious = tmp_path / "precious.tif"
    precious.write_bytes(b"do not delete")
    conn = init_connection()
    conn.execute("UPDATE scenes SET scl_path = ? WHERE scene_id = ?", (str(precious), NAME_B))
    conn.commit()
    conn.close()

    body = client.delete(f"/api/scenes/{NAME_B}").json()
    assert precious.read_bytes() == b"do not delete"
    assert "precious.tif" in body["files_failed"]


def test_delete_when_a_file_is_already_missing_still_succeeds(two_scenes):
    client, _, _ = two_scenes
    Path(f"data/cogs/{NAME_B}_scl.tif").unlink()
    resp = client.delete(f"/api/scenes/{NAME_B}")
    assert resp.status_code == 200 and resp.json()["files_failed"] == []
