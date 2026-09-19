"""IRIS Ingestion — Universal loader, capability detection, and COG conversion."""

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.warp import transform_bounds

logger = logging.getLogger("iris.ingestion")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@dataclass
class SceneMetadata:
    """Metadata extracted during capability detection."""
    scene_id: str
    sensor: str
    acquisition_date: str
    crs: str
    driver: str
    bands: int
    dtype: str
    nodata: Optional[float]
    bounds: list[float]  # [minx, miny, maxx, maxy] in native CRS
    bounds_wgs84: list[float]  # [min_lon, min_lat, max_lon, max_lat] in EPSG:4326
    resolution: tuple[float, float]
    width: int
    height: int
    raw_checksum: str


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 checksum of a file in chunks to bound memory usage."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def detect_sensor(file_name: str) -> str:
    """Detect sensor type from filename patterns."""
    name_upper = file_name.upper()
    if re.search(r"S2[AB]?|SENTINEL[-_]?2|MSIL[12][AC]", name_upper):
        return "sentinel2"
    if re.search(r"LC0[89]|LT0[45]|LE07|LANDSAT", name_upper):
        return "landsat8"
    if re.search(r"LISS[_-]?\d|CARTOSAT|BHUVAN", name_upper):
        return "bhuvan"
    return "unknown"


def detect_acquisition_date(file_name: str) -> str:
    """Extract acquisition date (YYYY-MM-DD) from filename patterns."""
    # Pattern 1: ISO date like YYYYMMDDTHHMMSS (e.g. S2A_MSIL2A_20240510T053651_...)
    match = re.search(r"(?:^|[^0-9])(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])T\d{6}", file_name)
    if match:
        year, month, day = match.group(1), match.group(2), match.group(3)
        return f"{year}-{month}-{day}"

    # Pattern 2: Hyphenated YYYY-MM-DD
    match = re.search(r"(?:^|[^0-9])(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])(?![0-9])", file_name)
    if match:
        year, month, day = match.group(1), match.group(2), match.group(3)
        return f"{year}-{month}-{day}"

    # Pattern 3: Compact YYYYMMDD
    match = re.search(r"(?:^|[^0-9])(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?![0-9])", file_name)
    if match:
        year, month, day = match.group(1), match.group(2), match.group(3)
        return f"{year}-{month}-{day}"

    return "unknown"


def inspect_raster(file_path: Path) -> SceneMetadata:
    """Perform capability detection on a satellite image file.
    
    Inspects format, CRS, bands, resolution, sensor, and acquisition date.
    """
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {file_path}")

    try:
        with rasterio.open(str(file_path)) as src:
            crs_str = src.crs.to_string() if src.crs else "unknown"
            bounds = [float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top)]
            res = (float(src.res[0]), float(src.res[1]))
            bands_count = src.count
            driver_name = src.driver
            dtype_name = src.dtypes[0] if src.dtypes else "unknown"
            nodata_val = float(src.nodata) if src.nodata is not None else None
            width = src.width
            height = src.height

            # Reproject bounds to WGS84 for MapLibre / WebMercator rendering
            bounds_wgs84 = None
            if src.crs:
                try:
                    w_bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                    bounds_wgs84 = [float(w_bounds[0]), float(w_bounds[1]), float(w_bounds[2]), float(w_bounds[3])]
                except Exception as ex:
                    logger.warning("Could not reproject bounds to WGS84: %s", ex)

            if not bounds_wgs84:
                if bounds[0] >= -180 and bounds[2] <= 180 and bounds[1] >= -90 and bounds[3] <= 90:
                    bounds_wgs84 = bounds
                else:
                    bounds_wgs84 = [78.9629 - 0.5, 20.5937 - 0.5, 78.9629 + 0.5, 20.5937 + 0.5]
    except RasterioIOError as e:
        raise ValueError(f"Failed to open raster file: {e}") from e
    except Exception as e:
        raise ValueError(f"Invalid raster file format: {e}") from e

    scene_id = re.sub(r"[^\w\-_.]", "_", file_path.stem)
    sensor = detect_sensor(file_path.name)
    acq_date = detect_acquisition_date(file_path.name)
    raw_checksum = compute_sha256(file_path)

    metadata = SceneMetadata(
        scene_id=scene_id,
        sensor=sensor,
        acquisition_date=acq_date,
        crs=crs_str,
        driver=driver_name,
        bands=bands_count,
        dtype=dtype_name,
        nodata=nodata_val,
        bounds=bounds,
        bounds_wgs84=bounds_wgs84,
        resolution=res,
        width=width,
        height=height,
        raw_checksum=raw_checksum,
    )

    logger.info(
        "Capability detection: scene_id=%s, sensor=%s, date=%s, crs=%s, bands=%d, res=%s, bounds_wgs84=%s",
        metadata.scene_id,
        metadata.sensor,
        metadata.acquisition_date,
        metadata.crs,
        metadata.bands,
        metadata.resolution,
        metadata.bounds_wgs84,
    )

    return metadata


def convert_to_cog(
    input_path: Path,
    metadata: SceneMetadata,
    output_dir: Path = Path("data/cogs"),
    staging_dir: Path = Path("data/staging"),
) -> Path:
    """Convert input raster to Cloud-Optimized GeoTIFF (COG).
    
    Uses windowed chunked copying, DEFLATE compression, 512x512 tiling,
    computes internal overviews, and applies atomic rename via staging.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    staging_file = staging_dir / f"{metadata.scene_id}.tif"
    final_file = output_dir / f"{metadata.scene_id}.tif"

    if staging_file.exists():
        staging_file.unlink()

    with rasterio.open(str(input_path)) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="DEFLATE",
        )

        with rasterio.open(str(staging_file), "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                for _, window in src.block_windows(band_idx):
                    window_data = src.read(band_idx, window=window)
                    dst.write(window_data, band_idx, window=window)

    # Build internal overviews on the staged GeoTIFF
    overview_factors = [2, 4, 8, 16]
    with rasterio.open(str(staging_file), "r+") as dst:
        dst.build_overviews(overview_factors, Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")

    # Atomic rename from staging to production cogs
    if final_file.exists():
        final_file.unlink()
    os.replace(staging_file, final_file)

    logger.info("COG created successfully: %s", final_file)
    return final_file
