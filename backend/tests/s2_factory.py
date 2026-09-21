"""Synthetic Sentinel-2 L2A products for tests.

Produces a real .SAFE folder layout (nested, as downloaded): MTD_MSIL2A.xml, GRANULE/*/IMG_DATA/R10m with B02/B03/
B04/B08, and R20m with the SCL. Imagery is a smooth random landscape so ECC has real texture to lock onto.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

from ingestion.loader import io_path

BASELINE_OFFSET = -1000.0  # processing baseline >= 04.00 adds -1000 to every DN

# Surface reflectance (B02, B03, B04, B08) of the land-cover kinds a `landcover` patch can paint. The default landscape
# is vegetation (NDVI ~0.6). "bare" and "built" both sit below NDVI 0.15 and are not water; "built" is the brighter
# of the two. NIR values are chosen so every transition between kinds is a clear B08 change.
LANDCOVER_REFLECTANCE = {
    "vegetation": (0.03, 0.06, 0.04, 0.40),  # NDVI 0.82
    "bare": (0.10, 0.13, 0.17, 0.22),  # NDVI 0.13
    "built": (0.20, 0.22, 0.30, 0.30),  # NDVI 0.0, and much brighter than bare soil
    "water": (0.06, 0.05, 0.03, 0.01),  # NDWI 0.67
}

MTD_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-2A_User_Product xmlns:n1="https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd">
  <n1:General_Info>
    <Product_Info>
      <PRODUCT_START_TIME>{date}T05:06:49.024Z</PRODUCT_START_TIME>
      <PRODUCT_URI>{name}.SAFE</PRODUCT_URI>
      <PROCESSING_LEVEL>Level-2A</PROCESSING_LEVEL>
      <PRODUCT_TYPE>S2MSI2A</PRODUCT_TYPE>
      <PROCESSING_BASELINE>05.12</PROCESSING_BASELINE>
    </Product_Info>
    <Product_Image_Characteristics>
      <BOA_ADD_OFFSET_VALUES_LIST>
{offsets}
      </BOA_ADD_OFFSET_VALUES_LIST>
    </Product_Image_Characteristics>
  </n1:General_Info>
</n1:Level-2A_User_Product>
"""


@dataclass
class Scenario:
    """Everything that differs between two acquisitions of the same place."""
    date: str = "2024-01-10"
    size: int = 512
    seed: int = 7  # the landscape; the same seed = the same ground
    noise_seed: int = 1  # per-acquisition sensor noise
    shift: Tuple[float, float] = (0.0, 0.0)  # (dx, dy) pixels the whole scene is displaced by
    gain: float = 1.0  # radiometric gain applied to all bands
    offset: float = 0.0  # radiometric offset (DN)
    patches: List[Tuple[int, int, int, int, float]] = field(default_factory=list)  # (r0, r1, c0, c1, NIR factor)
    landcover: List[Tuple[int, int, int, int, str]] = field(default_factory=list)  # (r0, r1, c0, c1, kind) repainted
    cloud: Optional[Tuple[int, int, int, int]] = None  # (r0, r1, c0, c1) painted as SCL class 9
    clouds: List[Tuple[int, int, int, int]] = field(default_factory=list)  # further clouds, same painting
    nodata_rows: int = 0  # bottom rows with no data (SCL 0)
    origin: Tuple[float, float] = (300000.0, 3100000.0)  # UTM upper-left; move it for a different grid
    crs: str = "EPSG:32643"


def _landscape(size: int, seed: int) -> np.ndarray:
    """(4, size, size) smooth, textured reflectance-like DNs for B02, B03, B04, B08."""
    rng = np.random.default_rng(seed)
    bands = []
    common = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), 5)
    for i, (mean, amp) in enumerate([(1500, 500), (1700, 600), (1600, 700), (3500, 1400)]):
        own = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), 7)
        field_ = 0.6 * (common - common.mean()) / common.std() + 0.4 * (own - own.mean()) / own.std()
        bands.append(mean + amp * field_)
    return np.stack(bands)


def render(scn: Scenario) -> Tuple[np.ndarray, np.ndarray]:
    """(bands uint16 (4,H,W), scl uint8 (H//2, W//2)) for a scenario."""
    size = scn.size
    bands = _landscape(size, scn.seed)

    for r0, r1, c0, c1, nir_factor in scn.patches:
        bands[3, r0:r1, c0:c1] *= nir_factor  # a NIR change, e.g. vegetation clearing (< 1) or growth (> 1)
        bands[2, r0:r1, c0:c1] *= 1.0 / max(nir_factor, 0.2) ** 0.5

    for r0, r1, c0, c1, kind in scn.landcover:
        # keep the landscape's texture (ECC needs it) around the kind's reflectance
        for i, refl in enumerate(LANDCOVER_REFLECTANCE[kind]):
            base = refl * 10000.0 - BASELINE_OFFSET
            bands[i, r0:r1, c0:c1] = base + 0.15 * (bands[i, r0:r1, c0:c1] - bands[i].mean())

    if scn.shift != (0.0, 0.0):
        m = np.array([[1, 0, -scn.shift[0]], [0, 1, -scn.shift[1]]], dtype=np.float32)
        bands = np.stack(
            [cv2.warpAffine(b, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for b in bands]
        )

    rng = np.random.default_rng(scn.noise_seed)
    bands = bands * scn.gain + scn.offset + rng.normal(0, 15, bands.shape)

    scl = np.full((size // 2, size // 2), 4, dtype=np.uint8)  # vegetation
    scl[: size // 8, :] = 5  # a strip of bare soil
    for r0, r1, c0, c1 in ([scn.cloud] if scn.cloud is not None else []) + list(scn.clouds):
        scl[r0 // 2 : r1 // 2, c0 // 2 : c1 // 2] = 9
        bands[:, r0:r1, c0:c1] = 9000 + rng.normal(0, 30, (4, r1 - r0, c1 - c0))  # bright cloud
    if scn.nodata_rows:
        scl[-(scn.nodata_rows // 2) :, :] = 0
        bands[:, -scn.nodata_rows :, :] = 0

    return np.clip(bands, 1, 65535).astype(np.uint16) * (bands > 0), scl


def _write(path: Path, data: np.ndarray, transform, crs: str, dtype: str) -> None:
    io_path(path.parent).mkdir(parents=True, exist_ok=True)
    count = 1 if data.ndim == 2 else data.shape[0]
    height, width = data.shape[-2:]
    # Real JPEG2000 where the GDAL build can write it. It must be lossless: JP2OpenJPEG defaults to lossy, which
    # would turn categorical SCL classes into values that were never in the scene.
    for driver, opts in (("JP2OpenJPEG", {"QUALITY": "100", "REVERSIBLE": "YES"}), ("GTiff", {})):
        try:
            with rasterio.open(
                str(path), "w", driver=driver, height=height, width=width, count=count, dtype=dtype, crs=crs,
                transform=transform, **opts,
            ) as dst:
                dst.write(data if data.ndim == 3 else data[None])
            return
        except Exception:
            io_path(path).unlink(missing_ok=True)
    raise RuntimeError("could not write test raster")


def make_safe(root: Path, name: str, scn: Scenario) -> Path:
    """Write <root>/<name>.SAFE/... and return the .SAFE folder."""
    safe = root / f"{name}.SAFE"
    granule = safe / "GRANULE" / "L2A_T43PHM_A000001_20240110T051600" / "IMG_DATA"
    bands, scl = render(scn)
    size = scn.size
    t10 = from_origin(scn.origin[0], scn.origin[1], 10.0, 10.0)
    t20 = from_origin(scn.origin[0], scn.origin[1], 20.0, 20.0)
    tile = "T43PHM"
    stamp = scn.date.replace("-", "") + "T050649"
    for band, arr in zip(("B02", "B03", "B04", "B08"), bands):
        _write(granule / "R10m" / f"{tile}_{stamp}_{band}_10m.jp2", arr, t10, scn.crs, "uint16")
    _write(granule / "R20m" / f"{tile}_{stamp}_SCL_20m.jp2", scl, t20, scn.crs, "uint8")

    offsets = "\n".join(f'        <BOA_ADD_OFFSET band_id="{i}">{int(BASELINE_OFFSET)}</BOA_ADD_OFFSET>' for i in range(13))
    io_path(safe / "MTD_MSIL2A.xml").write_text(MTD_TEMPLATE.format(date=scn.date, name=name, offsets=offsets), encoding="utf-8")
    return safe


def ingest_without_embedding(safe_or_file: Path) -> str:
    """Catalog a scene exactly as POST /api/ingest does, minus the (slow) embedding job. Returns the scene_id."""
    from catalog.database import init_connection, init_schema, upsert_scene
    from ingestion.safe import build_sentinel2_cogs, inspect_safe_product, open_safe_product

    product = open_safe_product(safe_or_file)
    assert product is not None
    metadata = inspect_safe_product(product)
    outputs = build_sentinel2_cogs(product)
    conn = init_connection()
    try:
        init_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        upsert_scene(
            conn,
            {
                "scene_id": metadata.scene_id,
                "sensor": metadata.sensor,
                "acquisition_date": metadata.acquisition_date,
                "crs": metadata.crs,
                "file_path": str(safe_or_file),
                "cog_path": f"data/cogs/{outputs.display_cog.name}",
                "analysis_cog_path": f"data/cogs/{outputs.analysis_cog.name}",
                "scl_path": f"data/cogs/{outputs.scl_cog.name}",
                "cloud_pct": outputs.cloud_pct,
                "raw_checksum": metadata.raw_checksum,
            },
            bounds_wgs84=metadata.bounds_wgs84,
        )
        conn.commit()
    finally:
        conn.close()
    return metadata.scene_id


# ---- Landsat Collection 2 Level-2 and generic rasters -------------------------------------------------------------------


LANDSAT_CLEAR_QA = 21824  # QA_PIXEL of a clear land pixel: bits 3 (cloud), 4 (shadow) and 5 (snow) all zero


def _write_gtiff(path: Path, data: np.ndarray, transform, crs: str) -> None:
    """A real GeoTIFF (Landsat ships GeoTIFFs; _write prefers JPEG2000, which would be wrong for a .TIF)."""
    io_path(path.parent).mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(path), "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype=str(data.dtype),
        crs=crs, transform=transform,
    ) as dst:
        dst.write(data, 1)


def make_landsat(root: Path, prefix: str, scn: Scenario, snow: Optional[Tuple[int, int, int, int]] = None) -> Path:
    """Write a Landsat C2 L2 scene folder <root>/<prefix>/ with SR_B2..B5 and QA_PIXEL (30 m, uint16) and return it.

    The scene is `scn` rendered like a Sentinel-2 scene, then re-expressed in Landsat's scaling (reflectance = DN * 2.75e-5 -
    0.2). The scenario's cloud / clouds become QA_PIXEL cloud bits (bit 3) and its no-data rows fill (bit 0); `snow` is a
    (r0, r1, c0, c1) box of snow bits (bit 5). The scenario's landcover, gain and so on apply as for Sentinel-2.
    """
    bands, scl = render(scn)  # Sentinel-2 DN (with the -1000 baseline offset), scl at half resolution
    reflectance = (bands.astype(np.float32) + BASELINE_OFFSET) / 10000.0
    raw = np.clip(np.rint((reflectance + 0.2) / 2.75e-5), 1, 65535).astype(np.uint16)
    raw[:, (bands == 0).all(axis=0)] = 0  # fill

    qa_full = np.full(bands.shape[1:], LANDSAT_CLEAR_QA, dtype=np.uint16)
    scl_full = np.kron(scl, np.ones((2, 2), dtype=np.uint8))
    qa_full[scl_full == 9] |= 1 << 3
    qa_full[scl_full == 3] |= 1 << 4
    qa_full[scl_full == 0] |= 1
    if snow is not None:
        r0, r1, c0, c1 = snow
        qa_full[r0:r1, c0:c1] |= 1 << 5

    folder = root / prefix
    transform = from_origin(scn.origin[0], scn.origin[1], 30.0, 30.0)
    for suffix, arr in zip(("SR_B2", "SR_B3", "SR_B4", "SR_B5"), raw):
        _write_gtiff(folder / f"{prefix}_{suffix}.TIF", arr, transform, scn.crs)
    _write_gtiff(folder / f"{prefix}_QA_PIXEL.TIF", qa_full, transform, scn.crs)
    return folder


def ingest_landsat_without_embedding(folder: Path) -> str:
    """Catalog a Landsat scene exactly as POST /api/ingest does, minus the embedding job. Returns the scene_id."""
    from catalog.database import init_connection, init_schema, upsert_scene
    from ingestion.landsat import build_landsat_cogs, inspect_landsat_product, open_landsat_product

    product = open_landsat_product(folder)
    assert product is not None
    metadata = inspect_landsat_product(product)
    outputs = build_landsat_cogs(product)
    conn = init_connection()
    try:
        init_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        upsert_scene(
            conn,
            {
                "scene_id": metadata.scene_id, "sensor": metadata.sensor, "acquisition_date": metadata.acquisition_date,
                "crs": metadata.crs, "file_path": str(folder), "cog_path": f"data/cogs/{outputs.display_cog.name}",
                "analysis_cog_path": f"data/cogs/{outputs.analysis_cog.name}", "scl_path": f"data/cogs/{outputs.scl_cog.name}",
                "cloud_pct": outputs.cloud_pct, "raw_checksum": metadata.raw_checksum,
            },
            bounds_wgs84=metadata.bounds_wgs84,
        )
        conn.commit()
    finally:
        conn.close()
    return metadata.scene_id


def make_generic(path: Path, data: np.ndarray, res: float = 30.0, crs: str = "EPSG:32643", origin=(300000.0, 3100000.0)) -> Path:
    """Write a plain multi-band GeoTIFF (bands, H, W) with no QA band at `res` metres."""
    io_path(path.parent).mkdir(parents=True, exist_ok=True)
    transform = from_origin(origin[0], origin[1], res, res)
    with rasterio.open(
        str(path), "w", driver="GTiff", height=data.shape[1], width=data.shape[2], count=data.shape[0],
        dtype=str(data.dtype), crs=crs, transform=transform, nodata=0,
    ) as dst:
        dst.write(data)
    return path


def ingest_generic_without_embedding(file_path: Path) -> str:
    """Catalog a generic raster as POST /api/ingest does (COG(s), heuristic cloud estimate), minus embedding."""
    from catalog.database import init_connection, init_schema, upsert_scene
    from ingestion.loader import convert_generic_to_cogs, heuristic_cloud_pct, inspect_raster

    metadata = inspect_raster(file_path)
    outputs = convert_generic_to_cogs(file_path, metadata)
    cloud_pct = heuristic_cloud_pct(file_path, metadata.sensor)
    conn = init_connection()
    try:
        init_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        upsert_scene(
            conn,
            {
                "scene_id": metadata.scene_id, "sensor": metadata.sensor, "acquisition_date": metadata.acquisition_date,
                "crs": metadata.crs, "file_path": str(file_path), "cog_path": f"data/cogs/{outputs.display_cog.name}",
                "analysis_cog_path": f"data/cogs/{outputs.analysis_cog.name}" if outputs.analysis_cog else None,
                "scl_path": None, "cloud_pct": cloud_pct, "raw_checksum": metadata.raw_checksum,
            },
            bounds_wgs84=metadata.bounds_wgs84,
        )
        conn.commit()
    finally:
        conn.close()
    return metadata.scene_id
