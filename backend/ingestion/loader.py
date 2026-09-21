"""IRIS Ingestion — Universal loader, capability detection, and COG conversion."""

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.warp import transform_bounds
from rasterio.windows import Window

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


_EXTENDED_PREFIX = "\\\\?\\"  # \\?\


def io_path(path: Path) -> Path:
    """Path form for Python file I/O that is safe beyond Windows' 260-character MAX_PATH.

    Python's open/stat/glob fail on longer paths unless LongPathsEnabled is set system-wide (it usually is not),
    and a nested Sentinel-2 .SAFE path is already ~170 characters before the analyst's own folders. The
    extended-length prefix lifts the limit for Python. GDAL handles long paths itself, so hand rasterio the
    plain form (plain_path). A no-op off Windows.
    """
    if os.name != "nt":
        return path
    text = str(path)
    if text.startswith(_EXTENDED_PREFIX):
        return path
    # The extended form needs an absolute, normalised path: no "/" separators, no "." or ".." components
    text = os.path.abspath(text)
    if text.startswith("\\\\"):  # UNC share: \\server\share -> \\?\UNC\server\share
        return Path(_EXTENDED_PREFIX + "UNC\\" + text[2:])
    return Path(_EXTENDED_PREFIX + text)


def plain_path(path: Path) -> Path:
    """Inverse of io_path: the ordinary form of a path, for GDAL, logging and anything shown to the analyst."""
    text = str(path)
    if os.name == "nt" and text.startswith(_EXTENDED_PREFIX):
        rest = text[len(_EXTENDED_PREFIX):]
        return Path("\\\\" + rest[4:]) if rest.startswith("UNC\\") else Path(rest)
    return path


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 checksum of a file in chunks to bound memory usage."""
    hasher = hashlib.sha256()
    with open(io_path(Path(file_path)), "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def detect_sensor(file_name: str) -> str:
    """Detect sensor type from filename patterns."""
    name_upper = file_name.upper()
    if re.search(r"S2[AB]?|SENTINEL[-_]?2|MSIL[12][AC]", name_upper):
        return "sentinel2"
    if "LC09" in name_upper:
        return "landsat9"
    if re.search(r"LC08|LT0[45]|LE07|LANDSAT", name_upper):
        return "landsat8"
    if re.search(r"LISS[_-]?(3|III)(?![0-9A-Z])", name_upper):
        return "liss3"
    if re.search(r"LISS[_-]?\d|CARTOSAT|BHUVAN", name_upper):
        return "bhuvan"
    return "unknown"


# LISS-III ground sampling distance is 23.5 m. A projected raster within this range with 3 or 4 bands is taken to be one.
LISS3_RESOLUTION_M = (21.0, 26.0)


def infer_sensor_from_raster(bands: int, resolution: tuple, projected: bool) -> str:
    """Sensor of a raster whose name says nothing: 3-4 bands at about 23 m is LISS-III, anything else is unknown.

    Resolution is only meaningful in a projected CRS (metres); in degrees it says nothing about the sensor.
    """
    if projected and bands in (3, 4) and LISS3_RESOLUTION_M[0] <= abs(resolution[0]) <= LISS3_RESOLUTION_M[1]:
        return "liss3"
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


def read_raster_bounds(file_path: Path) -> tuple[list[float], list[float]]:
    """Cheap bounds lookup for an already-converted COG: (native bounds, WGS84 bounds).

    Unlike inspect_raster this does no checksumming, so it is safe to call at app startup.
    """
    with rasterio.open(str(file_path)) as src:
        bounds = [float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top)]
        bounds_wgs84 = bounds
        if src.crs:
            w = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            bounds_wgs84 = [float(w[0]), float(w[1]), float(w[2]), float(w[3])]
    return bounds, bounds_wgs84


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
            projected = bool(src.crs and src.crs.is_projected)
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
    if sensor == "unknown":
        sensor = infer_sensor_from_raster(bands_count, res, projected)
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


# ---- generic rasters (Bhuvan / LISS-III / anything without a sensor-specific reader) -------------------------------------
#
# No QA band exists, so there is nothing to mask with. A 4+ band raster gets a heuristic cloud estimate that reduces the
# scene's CloudTrust (and through it the confidence of what is found there) but never removes a pixel; an RGB-only raster
# gets nothing at all.

HEURISTIC_NDVI_MAX = 0.2  # a bright pixel with NDVI below this is possibly cloud
HEURISTIC_BRIGHTNESS_PERCENTILE = 75.0
HEURISTIC_SAMPLE_PX = 2048  # the estimate is made on a copy decimated to at most this many pixels a side
DISPLAY_PERCENTILES = (2.0, 98.0)  # stretch of a non-8-bit generic raster, fixed per scene


def band_roles(sensor: str, count: int) -> Dict[str, Optional[int]]:
    """Zero-based band index of each role in a generic raster with a NIR band (count >= 4), else all None.

    LISS-III products are green, red, NIR, SWIR (no blue); every other 4+ band raster is taken as blue, green, red, NIR,
    the order change detection expects.
    """
    if count < 4:
        return {"blue": None, "green": None, "red": None, "nir": None}
    if sensor == "liss3":
        return {"blue": None, "green": 0, "red": 1, "nir": 2}
    return {"blue": 0, "green": 1, "red": 2, "nir": 3}


def heuristic_cloud_pct(file_path: Path, sensor: str) -> Optional[float]:
    """Share (0-100) of the scene's data pixels that are possibly cloud, or None when no estimate can be made.

    With a NIR band: a pixel is flagged when NDVI < 0.2 AND its visible brightness is above the scene's 75th percentile.
    Without one (RGB only) nothing can be flagged: CloudTrust stays 1.0 and a warning is logged.
    """
    with rasterio.open(str(file_path)) as src:
        roles = band_roles(sensor, src.count)
        if roles["nir"] is None:
            logger.warning("%s has no NIR band: no cloud masking is possible, CloudTrust = 1.0", file_path.name)
            return None
        scale = max(1, -(-max(src.width, src.height) // HEURISTIC_SAMPLE_PX))
        shape = (src.count, -(-src.height // scale), -(-src.width // scale))
        data = src.read(out_shape=shape).astype(np.float32)
        masks = src.read_masks(out_shape=shape) > 0
    valid = masks.all(axis=0) & ~(data == 0).all(axis=0)
    if not valid.any():
        return None
    red, nir = data[roles["red"]], data[roles["nir"]]
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(nir + red > 0, (nir - red) / (nir + red), np.nan)
    visible = [data[i] for i in (roles["blue"], roles["green"], roles["red"]) if i is not None]
    brightness = np.mean(visible, axis=0)
    threshold = np.percentile(brightness[valid], HEURISTIC_BRIGHTNESS_PERCENTILE)
    flagged = valid & (ndvi < HEURISTIC_NDVI_MAX) & (brightness > threshold)
    pct = round(100.0 * float(flagged.sum()) / float(valid.sum()), 2)
    logger.info("Heuristic masking applied — no QA band (%s: %.1f%% of data pixels flagged as possibly cloudy)", file_path.name, pct)
    return pct


@dataclass
class GenericOutputs:
    display_cog: Path
    analysis_cog: Optional[Path] = None  # only for a raster with a NIR band
    timings: Dict[str, float] = field(default_factory=dict)


def _stretch(dtype: str, sample: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Function mapping a band's raw values to 1..255: none needed for 8-bit, a percentile stretch fixed per scene else."""
    if dtype == "uint8":
        return lambda a: np.clip(a, 1, 255).astype(np.uint8)
    lo, hi = np.percentile(sample, DISPLAY_PERCENTILES) if sample.size else (0.0, 1.0)
    hi = hi if hi > lo else lo + 1.0
    return lambda a: np.clip((a.astype(np.float32) - lo) / (hi - lo) * 254.0 + 1.0, 1, 255).astype(np.uint8)


def convert_generic_to_cogs(
    input_path: Path,
    metadata: SceneMetadata,
    output_dir: Path = Path("data/cogs"),
    staging_dir: Path = Path("data/staging"),
) -> GenericOutputs:
    """COG(s) for a generic raster.

    Fewer than 4 bands (or a data type that cannot be re-expressed): one COG, a straight copy, as before. 4+ bands: an 8-bit
    RGB display COG (what the map and the embedding crops read, which take the first three bands as R, G, B) and an analysis COG
    in change detection's band order (blue, green, red, NIR; a LISS-III has no blue, so green stands in for it).
    """
    t0 = time.monotonic()
    roles = band_roles(metadata.sensor, metadata.bands)
    if roles["nir"] is None or metadata.dtype not in ("uint8", "uint16"):
        if roles["nir"] is not None:
            logger.warning("%s: %s data cannot be re-expressed for analysis; using a plain COG", input_path.name, metadata.dtype)
        cog = convert_to_cog(input_path, metadata, output_dir, staging_dir)
        return GenericOutputs(display_cog=cog, timings={"cog_s": round(time.monotonic() - t0, 2)})

    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "display": (staging_dir / f"{metadata.scene_id}.tif", output_dir / f"{metadata.scene_id}.tif"),
        "analysis": (staging_dir / f"{metadata.scene_id}_analysis.tif", output_dir / f"{metadata.scene_id}_analysis.tif"),
    }
    for staged, _ in targets.values():
        staged.unlink(missing_ok=True)

    order = [roles["blue"] if roles["blue"] is not None else roles["green"], roles["green"], roles["red"], roles["nir"]]
    with rasterio.open(str(input_path)) as src:
        scale = max(1, -(-max(src.width, src.height) // HEURISTIC_SAMPLE_PX))
        sample_shape = (3, -(-src.height // scale), -(-src.width // scale))
        sample = src.read([i + 1 for i in (roles["red"], roles["green"], order[0])], out_shape=sample_shape)
        stretch = _stretch(metadata.dtype, sample[sample > 0])
        base = {
            "driver": "GTiff", "width": src.width, "height": src.height, "crs": src.crs, "transform": src.transform,
            "tiled": True, "blockxsize": 512, "blockysize": 512, "compress": "DEFLATE", "BIGTIFF": "IF_SAFER", "nodata": 0,
        }
        with rasterio.open(str(targets["display"][0]), "w", count=3, dtype="uint8", **base) as display, rasterio.open(
            str(targets["analysis"][0]), "w", count=4, dtype=metadata.dtype, **base
        ) as analysis:
            for row in range(0, src.height, 1024):
                rows = min(1024, src.height - row)
                window = Window(0, row, src.width, rows)
                data = src.read(window=window)
                analysis.write(np.stack([data[i] for i in order]), window=window)
                rgb = np.stack([stretch(data[roles["red"]]), stretch(data[roles["green"]]), stretch(data[order[0]])])
                rgb[:, (data == 0).all(axis=0)] = 0
                display.write(rgb, window=window)
            analysis.update_tags(
                BANDS="blue,green,red,NIR", PRODUCT=metadata.scene_id, BOA_ADD_OFFSETS="0,0,0,0", SOURCE_SENSOR=metadata.sensor,
                NOTE="blue is the green band" if roles["blue"] is None else "",
            )
    for key in ("display", "analysis"):
        with rasterio.open(str(targets[key][0]), "r+") as dst:
            dst.build_overviews([2, 4, 8, 16], Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")
    for staged, final in targets.values():
        final.unlink(missing_ok=True)
        os.replace(staged, final)
    logger.info("Generic COGs written for %s (%s, %d bands)", metadata.scene_id, metadata.sensor, metadata.bands)
    return GenericOutputs(
        display_cog=targets["display"][1], analysis_cog=targets["analysis"][1], timings={"cog_s": round(time.monotonic() - t0, 2)}
    )
