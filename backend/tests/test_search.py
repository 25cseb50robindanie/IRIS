"""End-to-end integration tests for Semantic Search API."""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_bounds

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api.ingest as ingest_module
import embedding.embedder as embedder_module
from embedding.embedder import RemoteCLIPEmbedder
from embedding.index import get_vector_store
from main import app


def create_mock_imagery(
    file_path: Path,
    width: int = 672,
    height: int = 448,
    bands: int = 4,
    crs: str = "EPSG:32643",
    bounds: tuple[float, float, float, float] = (300000.0, 3100000.0, 306720.0, 3104480.0),
) -> Path:
    """Create synthetic satellite scene."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(*bounds, width, height)

    # Fill with realistic looking reflectance values
    data = (np.random.rand(bands, height, width) * 3000).astype(np.uint16)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": np.uint16,
        "crs": crs,
        "transform": transform,
        "nodata": 0,
    }

    with rasterio.open(str(file_path), "w", **profile) as dst:
        dst.write(data)

    return file_path


def test_search_api_e2e(tmp_path):
    """Test full flow: ingest scene -> index crops in FAISS -> search via POST /api/search."""
    client = TestClient(app)

    # 1. Create and ingest scene
    scene_file = tmp_path / "S2A_MSIL2A_20240510T053651_search_test.tif"
    create_mock_imagery(scene_file, width=672, height=448)

    ingest_resp = client.post("/api/ingest", json={"file_path": str(scene_file)})
    assert ingest_resp.status_code == 200
    ingest_data = ingest_resp.json()
    assert ingest_data["scene_id"] == "S2A_MSIL2A_20240510T053651_search_test"
    # Ingest returns as soon as the COG exists; embedding is reported through the status endpoint
    assert ingest_data["embedding_status"] == "queued"

    status_resp = client.get(f"/api/status/{ingest_data['scene_id']}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["state"] == "ready"
    assert status_data["tiles_count"] > 0
    assert status_data["crops_embedded"] == status_data["crops_total"] == status_data["tiles_count"]

    # 2. Perform search via POST /api/search
    search_resp = client.post(
        "/api/search",
        json={"query": "satellite view of airport runway and aircraft", "top_k": 5},
    )
    assert search_resp.status_code == 200
    search_data = search_resp.json()

    assert search_data["query"] == "satellite view of airport runway and aircraft"
    assert search_data["total_results"] > 0
    assert len(search_data["results"]) <= 5

    # Check structure of top result
    top_result = search_data["results"][0]
    assert "tile_id" in top_result
    assert "scene_id" in top_result
    assert "score" in top_result
    assert isinstance(top_result["score"], float)
    assert len(top_result["bounds"]) == 4  # [min_lon, min_lat, max_lon, max_lat]

    # Verify our ingested scene is present in results or top results
    scene_ids = [r["scene_id"] for r in search_data["results"]]
    assert len(scene_ids) > 0
    assert any("search_test" in sid or "S2A" in sid for sid in scene_ids)

    # Verify score is in descending order
    scores = [r["score"] for r in search_data["results"]]
    assert scores == sorted(scores, reverse=True)


def test_search_validation_and_errors():
    """Test validation on search query."""
    client = TestClient(app)

    # Empty query
    resp = client.post("/api/search", json={"query": "   ", "top_k": 10})
    assert resp.status_code == 400


def test_search_with_empty_index_is_a_clear_error():
    """An empty index must produce an actionable 409, not empty results or a 500."""
    client = TestClient(app)

    resp = client.post("/api/search", json={"query": "dense forest and vegetation", "top_k": 10})
    assert resp.status_code == 409
    assert "indexed" in resp.json()["detail"]


def test_search_without_model_checkpoint_is_503(monkeypatch):
    """A missing RemoteCLIP checkpoint is reported as 503 with the reason, never a bare 500."""
    unit = np.zeros((1, 512), dtype=np.float32)
    unit[0, 0] = 1.0
    get_vector_store().add_vectors(unit)

    monkeypatch.setattr(RemoteCLIPEmbedder, "_instance", None)
    monkeypatch.setattr(embedder_module, "DEFAULT_CHECKPOINT_PATHS", [])

    client = TestClient(app)
    resp = client.post("/api/search", json={"query": "dense forest", "top_k": 5})
    assert resp.status_code == 503
    assert "RemoteCLIP-ViT-B-32.pt" in resp.json()["detail"]


def test_status_unknown_scene_is_404():
    client = TestClient(app)
    assert client.get("/api/status/no_such_scene").status_code == 404


def test_missing_checkpoint_fails_embedding_but_scene_stays_viewable(tmp_path, monkeypatch):
    """Ingest still succeeds (map loads); the embedding job records the missing-checkpoint error."""
    monkeypatch.setattr(ingest_module, "find_checkpoint", lambda *a, **k: None)
    client = TestClient(app)

    scene_file = tmp_path / "S2A_MSIL2A_20240510T053651_nockpt.tif"
    create_mock_imagery(scene_file, width=448, height=224)

    resp = client.post("/api/ingest", json={"file_path": str(scene_file)})
    assert resp.status_code == 200

    status_data = client.get(f"/api/status/{resp.json()['scene_id']}").json()
    assert status_data["state"] == "failed"
    assert "RemoteCLIP-ViT-B-32.pt" in status_data["error"]
    assert status_data["tiles_count"] == 0
