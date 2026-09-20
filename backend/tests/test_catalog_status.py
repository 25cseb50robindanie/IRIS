"""Tests for startup-resume state (GET /api/catalog/status) and FAISS persistence across restarts."""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import embedding.index as index_module
from embedding.index import FaissVectorStore, get_vector_store
from main import app


def make_scene(path: Path, width: int = 448, height: int = 448) -> Path:
    data = np.random.randint(1, 255, (3, height, width), dtype=np.uint8)
    with rasterio.open(
        str(path), "w", driver="GTiff", height=height, width=width, count=3, dtype="uint8",
        crs="EPSG:32643", transform=from_bounds(300000, 3100000, 304480, 3104480, width, height), nodata=0,
    ) as dst:
        dst.write(data)
    return path


def test_empty_catalog_status():
    resp = TestClient(app).get("/api/catalog/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_scenes"] is False
    assert data["scene_count"] == 0
    assert data["faiss_vectors"] == 0
    assert data["latest_scene"] is None


def test_status_after_ingest_returns_resumable_scene(tmp_path):
    client = TestClient(app)
    scene = make_scene(tmp_path / "S2A_MSIL2A_20240510T053651_first.tif")
    assert client.post("/api/ingest", json={"file_path": str(scene)}).status_code == 200

    data = client.get("/api/catalog/status").json()
    assert data["has_scenes"] is True
    assert data["scene_count"] == 1
    assert data["faiss_vectors"] > 0
    assert data["tiles_count"] == data["faiss_vectors"]

    latest = data["latest_scene"]
    assert latest["scene_id"] == "S2A_MSIL2A_20240510T053651_first"
    assert latest["crs"] == "EPSG:32643"
    assert len(latest["bounds_wgs84"]) == 4
    assert latest["bounds_wgs84"][0] < latest["bounds_wgs84"][2]
    assert latest["cog_url"].startswith("file:///")
    assert latest["tiles_count"] == data["tiles_count"]


def test_latest_scene_is_most_recent_import_including_reimport(tmp_path):
    client = TestClient(app)
    first = make_scene(tmp_path / "S2A_MSIL2A_20240510T053651_first.tif")
    second = make_scene(tmp_path / "S2A_MSIL2A_20240511T053651_second.tif")

    client.post("/api/ingest", json={"file_path": str(first)})
    client.post("/api/ingest", json={"file_path": str(second)})
    assert client.get("/api/catalog/status").json()["latest_scene"]["scene_id"].endswith("_second")

    client.post("/api/ingest", json={"file_path": str(first)})
    data = client.get("/api/catalog/status").json()
    assert data["scene_count"] == 2
    assert data["latest_scene"]["scene_id"].endswith("_first")


def test_scene_with_missing_cog_is_not_offered_for_resume(tmp_path):
    client = TestClient(app)
    scene = make_scene(tmp_path / "S2A_MSIL2A_20240510T053651_gone.tif")
    client.post("/api/ingest", json={"file_path": str(scene)})
    for cog in Path("data/cogs").glob("*.tif"):
        cog.unlink()

    data = client.get("/api/catalog/status").json()
    assert data["has_scenes"] is True
    assert data["latest_scene"] is None


def test_faiss_index_is_loaded_from_disk_after_restart():
    """A fresh process (empty store cache) must pick up the vectors saved by the previous one."""
    unit = np.eye(512, dtype=np.float32)[:3]
    get_vector_store().add_vectors(unit)
    assert Path("data/faiss_index.bin").exists()

    index_module._stores.clear()  # what a process restart does to the in-memory cache
    assert get_vector_store().total_vectors == 3

    resp = TestClient(app).get("/api/catalog/status")
    assert resp.json()["faiss_vectors"] == 3


def test_unreadable_faiss_index_is_set_aside_not_overwritten(tmp_path):
    bad = tmp_path / "faiss_index.bin"
    bad.write_bytes(b"not a faiss index")

    store = FaissVectorStore(index_path=bad)
    assert store.total_vectors == 0
    assert not bad.exists()
    assert len(list(tmp_path.glob("faiss_index.bin.corrupt-*"))) == 1
