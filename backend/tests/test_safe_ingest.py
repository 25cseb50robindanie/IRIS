"""Tests for Part A: Sentinel-2 L2A .SAFE ingestion."""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog.database import get_scene, init_connection, init_schema
from ingestion.loader import io_path, plain_path
from ingestion.safe import build_sentinel2_cogs, find_safe_root, inspect_safe_product, open_safe_product
from main import app
from s2_factory import Scenario, make_safe, render

NAME = "S2B_MSIL2A_20240110T050649_N0510_R019_T43PHM_20240110T090000"


def test_open_product_reads_metadata_and_bands(tmp_path):
    safe = make_safe(tmp_path, NAME, Scenario())
    product = open_safe_product(safe)

    assert product is not None
    assert product.scene_id == NAME
    assert product.acquisition_date == "2024-01-10"
    assert product.processing_baseline == "05.12"
    assert set(product.band_paths) == {"B02", "B03", "B04", "B08"}
    assert product.scl_path.name.endswith("_SCL_20m.jp2")
    assert product.add_offsets["B04"] == -1000.0


def test_folder_containing_a_safe_is_accepted(tmp_path):
    """A downloaded product is often nested (X.SAFE/X.SAFE); a parent folder must be enough."""
    inner = tmp_path / "download" / "wrapper"
    safe = make_safe(inner, NAME, Scenario())
    assert find_safe_root(tmp_path / "download") == safe.resolve()
    assert open_safe_product(tmp_path / "download").scene_id == NAME


def test_non_safe_folder_returns_none(tmp_path):
    (tmp_path / "just_a_folder").mkdir()
    assert open_safe_product(tmp_path / "just_a_folder") is None


def test_level_1c_product_is_rejected(tmp_path):
    l1c = tmp_path / "S2A_MSIL1C_x.SAFE"
    l1c.mkdir()
    (l1c / "MTD_MSIL1C.xml").write_text("<x/>")
    with pytest.raises(ValueError, match="Level-1C"):
        open_safe_product(l1c)


def test_missing_band_is_a_clear_error(tmp_path):
    safe = make_safe(tmp_path, NAME, Scenario())
    next(safe.rglob("*_B08_10m.jp2")).unlink()
    with pytest.raises(ValueError, match="B08"):
        open_safe_product(safe)


def test_two_products_in_one_folder_are_ambiguous(tmp_path):
    make_safe(tmp_path, NAME, Scenario())
    make_safe(tmp_path, NAME.replace("20240110", "20240220"), Scenario(date="2024-02-20"))
    with pytest.raises(ValueError, match="More than one"):
        find_safe_root(tmp_path)


def test_cogs_have_the_expected_bands_and_scl_stays_categorical(tmp_path):
    scn = Scenario(cloud=(64, 192, 64, 192), size=512)
    safe = make_safe(tmp_path, NAME, scn)
    product = open_safe_product(safe)
    out = build_sentinel2_cogs(product)

    with rasterio.open(out.analysis_cog) as src:
        assert (src.count, src.dtypes[0], src.width, src.height) == (4, "uint16", 512, 512)
        assert [src.descriptions[i] for i in range(4)] == ["B02", "B03", "B04", "B08"]
        assert src.overviews(1)  # a real COG has overviews
    with rasterio.open(out.display_cog) as src:
        assert (src.count, src.dtypes[0]) == (3, "uint8")

    with rasterio.open(out.scl_cog) as src:
        scl10 = src.read(1)
    _, scl20 = render(scn)
    assert scl10.shape == (512, 512)
    # nearest-neighbour: each 20 m pixel becomes an exact 2x2 block, and no new class values appear
    assert np.array_equal(scl10, np.kron(scl20, np.ones((2, 2), dtype=np.uint8)))
    assert set(np.unique(scl10)) <= set(np.unique(scl20))

    # 128x128 px cloud (class 9) of a 512x512 scene = 6.25%
    assert out.cloud_pct == pytest.approx(6.25, abs=0.01)


def test_cloud_percentage_counts_shadow_but_not_water_or_snow(tmp_path):
    """SCL 3 (shadow), 8/9 (cloud) and 10 (cirrus) count; 6 is WATER and 11 is snow."""
    from ingestion import safe as safe_module

    assert set(safe_module.SCL_CLOUD_CLASSES) == {3, 8, 9, 10}
    assert 6 not in safe_module.SCL_CLOUD_CLASSES and safe_module.SCL_SNOW == 11


def test_ingest_api_accepts_a_safe_folder(tmp_path):
    client = TestClient(app)
    safe = make_safe(tmp_path, NAME, Scenario(cloud=(0, 128, 0, 128)))

    resp = client.post("/api/ingest", json={"file_path": str(safe)})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["product_type"] == "sentinel2_safe"
    assert data["scene_id"] == NAME
    assert data["sensor"] == "sentinel2"
    assert data["acquisition_date"] == "2024-01-10"
    assert data["analysis_cog_path"].endswith(f"{NAME}_analysis.tif")
    assert data["scl_path"].endswith(f"{NAME}_scl.tif")
    assert data["cloud_pct"] == pytest.approx(100 * 128 * 128 / (512 * 512), abs=0.01)

    conn = init_connection()
    try:
        init_schema(conn)
        row = get_scene(conn, NAME)
    finally:
        conn.close()
    assert row["analysis_cog_path"] == data["analysis_cog_path"]
    assert row["scl_path"] == data["scl_path"]
    assert row["cloud_pct"] == pytest.approx(data["cloud_pct"])
    for key in ("analysis_cog_path", "scl_path", "cog_path"):
        assert Path(row[key]).is_file()


def test_ingest_api_rejects_a_folder_that_is_not_a_product(tmp_path):
    client = TestClient(app)
    (tmp_path / "plain").mkdir()
    resp = client.post("/api/ingest", json={"file_path": str(tmp_path / "plain")})
    assert resp.status_code == 400
    assert "Sentinel-2" in resp.json()["detail"]


def test_single_file_flow_still_reports_no_analysis_rasters(tmp_path):
    """The original GeoTIFF/TCI path must keep working unchanged."""
    from rasterio.transform import from_origin

    tif = tmp_path / "S2A_MSIL2A_20240110T053651_plain.tif"
    data = np.random.randint(1, 255, (3, 256, 256), dtype=np.uint8)
    with rasterio.open(
        tif, "w", driver="GTiff", height=256, width=256, count=3, dtype="uint8", crs="EPSG:32643",
        transform=from_origin(300000, 3100000, 10, 10), nodata=0,
    ) as dst:
        dst.write(data)

    resp = TestClient(app).post("/api/ingest", json={"file_path": str(tif)})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["product_type"] == "single_file"
    assert body["analysis_cog_path"] is None and body["scl_path"] is None and body["cloud_pct"] is None


def test_product_deeper_than_windows_max_path_is_ingested(tmp_path):
    """A nested .SAFE under an ordinary user folder passes 260 characters; Python's own I/O fails there."""
    deep = tmp_path
    for part in ("downloads_from_copernicus_2024", "sentinel2_l2a_products_for_review", "tile_43PHM_first_acquisition"):
        deep = deep / part
    safe = make_safe(deep, NAME, Scenario())

    longest = max(len(str(plain_path(p))) for p in io_path(safe).rglob("*.jp2"))  # plain rglob cannot see them
    assert longest > 260, f"fixture is not deep enough to exercise the limit ({longest})"

    product = open_safe_product(safe)
    assert product is not None and product.scene_id == NAME
    assert not str(product.scl_path).startswith("\\\\?\\")  # the analyst-facing paths stay ordinary

    metadata = inspect_safe_product(product)  # checksums every source file through the long-path form
    assert len(metadata.raw_checksum) == 64
    out = build_sentinel2_cogs(product)
    assert out.analysis_cog.is_file() and out.scl_cog.is_file()

    resp = TestClient(app).post("/api/ingest", json={"file_path": str(deep)})
    assert resp.status_code == 200, resp.text
    assert resp.json()["scene_id"] == NAME
