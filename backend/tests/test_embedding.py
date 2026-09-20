"""Tests for IRIS Embedding module: Crop Tiling, RemoteCLIP Embedder, and FAISS Vector Store."""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog.database import init_connection, init_schema, insert_tiles_batch, get_tiles_by_faiss_ids
from embedding.crop import extract_crops, normalize_to_rgb8
from embedding.embedder import RemoteCLIPEmbedder
from embedding.index import FaissVectorStore


def create_sample_scene(
    file_path: Path,
    width: int = 672,
    height: int = 448,
    bands: int = 4,
    crs: str = "EPSG:32643",
    bounds: tuple[float, float, float, float] = (300000.0, 3100000.0, 306720.0, 3104480.0),
) -> Path:
    """Create synthetic satellite scene with distinct quadrants to test tiling."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(*bounds, width, height)

    data = np.zeros((bands, height, width), dtype=np.uint16)
    # Left side: bright reflectance (~2000)
    data[:, :, : width // 2] = 2000
    # Right side: partial nodata (0) on bottom-right
    data[:, : height // 2, width // 2 :] = 1500
    # Bottom right corner is 0 (nodata/black)
    data[:, height // 2 :, width // 2 :] = 0

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


def test_crop_tiling_and_nodata_filtering(tmp_path):
    """Test extracting 224x224 crops and filtering out >50% nodata crops."""
    scene_file = tmp_path / "test_scene.tif"
    create_sample_scene(scene_file, width=672, height=448)  # 3 cols x 2 rows = 6 possible crops
    
    crops_dir = tmp_path / "crops"
    crops = extract_crops(
        cog_path=scene_file,
        scene_id="test_scene",
        crop_size=224,
        min_valid_fraction=0.5,
        output_dir=crops_dir,
    )

    # 6 possible crops:
    # (col 0, row 0): valid (100%)
    # (col 1, row 0): valid (100%)
    # (col 2, row 0): valid (100%)
    # (col 0, row 1): valid (100%)
    # (col 1, row 1): boundary / valid
    # (col 2, row 1): bottom-right is all 0s (0% valid) -> filtered out!
    assert len(crops) >= 4
    for c in crops:
        assert c.valid_pixel_frac >= 0.5
        assert c.file_path.exists()
        assert len(c.bounds_native) == 4
        assert len(c.bounds_wgs84) == 4


def test_remoteclip_embedder():
    """Test RemoteCLIP image and text encoding."""
    embedder = RemoteCLIPEmbedder.get_instance()

    # Test text embedding
    queries = ["satellite view of river", "airport runway with planes"]
    text_emb = embedder.embed_text(queries)
    assert text_emb.shape == (2, 512)
    assert text_emb.dtype == np.float32
    # Verify L2 normalization: norm should be ~1.0
    norms = np.linalg.norm(text_emb, axis=-1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)
    assert not np.isnan(text_emb).any()

    # Test image embedding
    dummy_imgs = [
        Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8)),
        Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8)),
    ]
    img_emb = embedder.embed_images(dummy_imgs, batch_size=2)
    assert img_emb.shape == (2, 512)
    assert img_emb.dtype == np.float32
    norms = np.linalg.norm(img_emb, axis=-1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)
    assert not np.isnan(img_emb).any()


def test_faiss_vector_store(tmp_path):
    """Test FAISS vector store insertion, persistence, and cosine search."""
    index_file = tmp_path / "faiss_index.bin"
    store = FaissVectorStore(index_path=index_file)
    assert store.total_vectors == 0

    # Create synthetic unit vectors
    np.random.seed(42)
    v1 = np.random.randn(1, 512).astype(np.float32)
    v1 = v1 / np.linalg.norm(v1)

    v2 = np.random.randn(1, 512).astype(np.float32)
    v2 = v2 / np.linalg.norm(v2)

    vectors = np.vstack([v1, v2])
    faiss_ids = store.add_vectors(vectors)
    assert faiss_ids == [0, 1]
    assert store.total_vectors == 2
    assert index_file.exists()

    # Test search with exact v1
    scores, indices = store.search(v1, top_k=2)
    assert indices[0][0] == 0
    assert abs(scores[0][0] - 1.0) < 1e-4

    # Test reload from disk
    store_reloaded = FaissVectorStore(index_path=index_file)
    assert store_reloaded.total_vectors == 2
    scores_r, indices_r = store_reloaded.search(v2, top_k=1)
    assert indices_r[0][0] == 1
    assert abs(scores_r[0][0] - 1.0) < 1e-4


def test_faiss_rejects_non_finite_vectors(tmp_path):
    """A NaN vector must never enter the HNSW graph."""
    store = FaissVectorStore(index_path=tmp_path / "faiss_index.bin")
    bad = np.full((1, 512), np.nan, dtype=np.float32)
    with pytest.raises(ValueError):
        store.add_vectors(bad)
    assert store.total_vectors == 0


def test_embedding_job_reports_stages_in_order(tmp_path):
    """Tiling -> embedding -> indexing -> ready, with live crop counts along the way."""
    import api.ingest as ingest_module
    from embedding.progress import embedding_progress

    scene_file = tmp_path / "test_scene.tif"
    create_sample_scene(scene_file, width=672, height=448)

    seen = []
    real_update = embedding_progress.update

    def spy(scene_id, **fields):
        seen.append(dict(fields))
        real_update(scene_id, **fields)

    # tiles.scene_id is a foreign key, so the scene row must exist before its tiles do
    conn = init_connection()
    init_schema(conn)
    conn.execute(
        "INSERT INTO scenes (scene_id, sensor, acquisition_date, crs, file_path) VALUES (?, ?, ?, ?, ?)",
        ("stage_scene", "Sentinel-2", "2024-05-10", "EPSG:32643", str(scene_file)),
    )
    conn.commit()
    conn.close()

    embedding_progress.start("stage_scene")
    embedding_progress.update = spy
    try:
        ingest_module._embed_scene(scene_file, "stage_scene")
    finally:
        embedding_progress.update = real_update

    states = [f["state"] for f in seen if "state" in f]
    assert states == ["tiling", "embedding", "indexing", "ready"]
    assert any("crops_generated" in f for f in seen)
    assert any(f.get("crops_embedded") for f in seen)

    snap = embedding_progress.snapshot("stage_scene")
    assert snap["state"] == "ready"
    assert snap["tiles_count"] == snap["crops_total"] > 0
    assert snap["device"] in ("cpu", "cuda")
