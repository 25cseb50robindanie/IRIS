"""Comprehensive test suite for IRIS Backend Ingestion and Tile Serving."""

import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_bounds

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog.database import init_connection, init_schema
from ingestion.loader import (
    compute_sha256,
    convert_to_cog,
    detect_acquisition_date,
    detect_sensor,
    inspect_raster,
)
from main import app


def create_sample_geotiff(
    file_path: Path,
    width: int = 1024,
    height: int = 1024,
    bands: int = 4,
    crs: str = "EPSG:32643",
    bounds: tuple[float, float, float, float] = (300000.0, 3100000.0, 310240.0, 3110240.0),
) -> Path:
    """Create a synthetic multi-band GeoTIFF for testing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(*bounds, width, height)
    
    # Generate mock satellite band data
    data = (np.random.rand(bands, height, width) * 4000).astype(np.uint16)
    
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


def test_sensor_and_date_detection():
    """Test sensor and acquisition date parsing logic."""
    s2_name = "S2A_MSIL2A_20240510T053651_N0510_R019_T43QHV_20240510T090847.tif"
    assert detect_sensor(s2_name) == "sentinel2"
    assert detect_acquisition_date(s2_name) == "2024-05-10"

    landsat_name = "LC08_L2SP_146039_20230615_20230622_02_T1.tif"
    assert detect_sensor(landsat_name) == "landsat8"
    assert detect_acquisition_date(landsat_name) == "2023-06-15"

    bhuvan_name = "LISS3_2022-11-20_Scene01.tif"
    assert detect_sensor(bhuvan_name) == "bhuvan"
    assert detect_acquisition_date(bhuvan_name) == "2022-11-20"

    unknown_name = "random_aerial_image.tif"
    assert detect_sensor(unknown_name) == "unknown"
    assert detect_acquisition_date(unknown_name) == "unknown"


def test_capability_detection_and_cog_conversion(tmp_path):
    """Test raster capability detection and COG generation."""
    sample_file = tmp_path / "S2A_MSIL2A_20240510T053651_test.tif"
    create_sample_geotiff(sample_file, width=1024, height=1024, bands=4)
    
    # 1. Capability detection
    metadata = inspect_raster(sample_file)
    assert metadata.scene_id == "S2A_MSIL2A_20240510T053651_test"
    assert metadata.sensor == "sentinel2"
    assert metadata.acquisition_date == "2024-05-10"
    assert metadata.bands == 4
    assert metadata.crs == "EPSG:32643"
    assert len(metadata.bounds) == 4
    assert metadata.bounds_wgs84 is not None
    assert len(metadata.bounds_wgs84) == 4
    assert len(metadata.raw_checksum) == 64

    # 2. COG Conversion
    cogs_dir = tmp_path / "cogs"
    staging_dir = tmp_path / "staging"
    cog_path = convert_to_cog(sample_file, metadata, output_dir=cogs_dir, staging_dir=staging_dir)

    assert cog_path.exists()
    assert not (staging_dir / f"{metadata.scene_id}.tif").exists()  # Atomic rename succeeded

    # 3. Verify COG properties
    with rasterio.open(str(cog_path)) as dst:
        assert dst.profile["driver"] == "GTiff"
        assert dst.profile["tiled"] is True
        assert dst.profile["blockxsize"] == 512
        assert dst.profile["blockysize"] == 512
        assert dst.profile["compress"].upper() == "DEFLATE"
        # Check overviews
        overviews = dst.overviews(1)
        assert overviews == [2, 4, 8, 16]


def test_api_ingest_and_tile_serving(tmp_path):
    """Test full HTTP API flow: ingestion + database cataloging + titiler tile serving."""
    client = TestClient(app)

    # 1. Test Health endpoint
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json() == {"status": "ok"}

    # 2. Create mock scene
    sample_file = tmp_path / "S2A_MSIL2A_20240510T053651_integration.tif"
    create_sample_geotiff(sample_file, width=1024, height=1024, bands=4)

    # 3. Ingest scene via POST /api/ingest
    resp = client.post("/api/ingest", json={"file_path": str(sample_file)})
    assert resp.status_code == 200
    data = resp.json()

    assert data["scene_id"] == "S2A_MSIL2A_20240510T053651_integration"
    assert data["crs"] == "EPSG:32643"
    assert "data/cogs/" in data["cog_path"]
    assert "file://" in data["cog_url"]
    assert len(data["bounds"]) == 4
    assert len(data["bounds_wgs84"]) == 4

    # 4. Verify catalog database entry
    conn = init_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT scene_id, sensor, acquisition_date, crs, file_path, cog_path, raw_checksum FROM scenes WHERE scene_id = ?", (data["scene_id"],))
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == data["scene_id"]
    assert row[1] == "sentinel2"
    assert row[2] == "2024-05-10"
    assert row[3] == "EPSG:32643"
    assert row[4] == str(sample_file)
    assert row[5] == data["cog_path"]
    assert len(row[6]) == 64

    # 5. Verify Titiler Tile Serving
    # Test titiler metadata endpoint for the COG using cog_url
    # cog_url is raw; encode it once as a query value (like encodeURIComponent)
    cog_url = data["cog_url"]
    encoded_url = quote(cog_url, safe="")
    info_resp = client.get(f"/tiles/cog/info?url={encoded_url}")
    assert info_resp.status_code == 200
    info_json = info_resp.json()
    assert info_json["band_metadata"] is not None

    # Test titiler tilejson endpoint, as consumed by MapView
    tile_resp = client.get(f"/tiles/cog/tilejson.json?url={encoded_url}")
    assert tile_resp.status_code == 200
    tilejson = tile_resp.json()
    assert tilejson["tiles"]
    # titiler is mounted at /tiles; the advertised tile URLs must include that prefix
    assert "/tiles/cog/tiles/" in tilejson["tiles"][0]
    assert len(tilejson["bounds"]) == 4


def test_tilejson_with_special_characters_in_filename(tmp_path):
    """A COG whose filename has spaces/special characters must be servable via tilejson."""
    client = TestClient(app)

    sample_file = tmp_path / "S2A_MSIL2A_20240511T053651 special #1 (test).tif"
    create_sample_geotiff(sample_file, width=1024, height=1024, bands=4)

    resp = client.post("/api/ingest", json={"file_path": str(sample_file)})
    assert resp.status_code == 200
    cog_url = resp.json()["cog_url"]
    assert "%" not in cog_url

    tile_resp = client.get(f"/tiles/cog/tilejson.json?url={quote(cog_url, safe='')}")
    assert tile_resp.status_code == 200
    assert len(tile_resp.json()["bounds"]) == 4


def test_ingest_accepts_file_uri(tmp_path):
    """POST /api/ingest accepts a file:/// URI as well as a plain path."""
    client = TestClient(app)

    sample_file = tmp_path / "S2A_MSIL2A_20240512T053651_uri.tif"
    create_sample_geotiff(sample_file, width=1024, height=1024, bands=4)

    resp = client.post("/api/ingest", json={"file_path": sample_file.as_uri()})
    assert resp.status_code == 200
    assert resp.json()["scene_id"] == "S2A_MSIL2A_20240512T053651_uri"


def test_api_ingest_error_handling():
    """Test error handling for missing or invalid files."""
    client = TestClient(app)

    # Missing file
    resp = client.post("/api/ingest", json={"file_path": "non_existent_file_xyz_123.tif"})
    assert resp.status_code == 404

    # Empty path
    resp = client.post("/api/ingest", json={"file_path": "   "})
    assert resp.status_code == 400


if __name__ == "__main__":
    pytest.main(["-v", __file__])
