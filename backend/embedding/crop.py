"""IRIS Crop Tiler — Extract 224x224 RGB crops from satellite COG."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from rasterio.warp import transform_bounds

logger = logging.getLogger("iris.embedding.crop")


@dataclass
class CropInfo:
    """Metadata for an extracted satellite imagery crop."""
    tile_id: str
    scene_id: str
    col_off: int
    row_off: int
    file_path: Path
    bounds_native: List[float]  # [minx, miny, maxx, maxy] in native CRS
    bounds_wgs84: List[float]   # [min_lon, min_lat, max_lon, max_lat] in EPSG:4326
    valid_pixel_frac: float


def normalize_to_rgb8(arr: np.ndarray, nodata: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Convert multi-band raster array to uint8 RGB (H, W, 3) and compute valid pixel mask.
    
    Returns:
        rgb_uint8: np.ndarray of shape (H, W, 3), dtype uint8
        valid_mask: np.ndarray of shape (H, W), dtype bool (True where pixel is valid)
    """
    # arr shape is (C, H, W)
    channels, height, width = arr.shape
    
    if channels >= 3:
        rgb = arr[:3].astype(np.float32)
    elif channels == 1:
        rgb = np.repeat(arr.astype(np.float32), 3, axis=0)
    else:
        # 2 channels: repeat 1st channel for 3rd
        rgb = np.stack([arr[0], arr[1], arr[0]], axis=0).astype(np.float32)

    # Determine valid mask
    valid_mask = np.ones((height, width), dtype=bool)
    
    if nodata is not None:
        for c in range(rgb.shape[0]):
            valid_mask &= (rgb[c] != nodata)
            
    # Check for NaNs
    valid_mask &= ~np.isnan(rgb).any(axis=0)
    
    # Check for all-black pixels (nodata / background borders)
    all_zero = (rgb == 0).all(axis=0)
    valid_mask &= ~all_zero

    # Scale to 0..255
    # If standard 0-255 uint8 range
    max_val = np.nanmax(rgb) if np.any(valid_mask) else 0.0
    if max_val > 255.0:
        # Typical 12-bit / 16-bit satellite reflectance (0..10000 range, surface reflectance typically < 3500)
        # Use a robust 2% to 98% percentile stretch over valid pixels or standard 3500 scale
        if np.count_nonzero(valid_mask) > 10:
            valid_pixels = rgb[:, valid_mask]
            p2 = np.percentile(valid_pixels, 2)
            p98 = np.percentile(valid_pixels, 98)
            if p98 > p2:
                rgb = np.clip((rgb - p2) / (p98 - p2) * 255.0, 0, 255)
            else:
                rgb = np.clip(rgb / 3500.0 * 255.0, 0, 255)
        else:
            rgb = np.clip(rgb / 3500.0 * 255.0, 0, 255)
    else:
        rgb = np.clip(rgb, 0, 255)

    # Fill invalid pixels with 0 (neutral black) so no NaNs propagate
    rgb[:, ~valid_mask] = 0

    rgb_uint8 = np.transpose(rgb.astype(np.uint8), (1, 2, 0))  # (H, W, 3)
    return rgb_uint8, valid_mask


def extract_crops(
    cog_path: Path,
    scene_id: str,
    crop_size: int = 224,
    min_valid_fraction: float = 0.5,
    output_dir: Path = Path("data/crops"),
    on_progress: Optional[Callable[[int], None]] = None,
) -> List[CropInfo]:
    """Extract 224x224 RGB crops from a COG and save valid ones to disk.

    Skips crops that have less than min_valid_fraction valid pixels (more than 50% nodata/black).
    on_progress, if given, is called with the running count of crops kept after each one is saved.
    """
    cog_path = Path(cog_path)
    if not cog_path.exists():
        raise FileNotFoundError(f"COG file not found: {cog_path}")

    scene_crops_dir = output_dir / scene_id
    scene_crops_dir.mkdir(parents=True, exist_ok=True)

    crops_info: List[CropInfo] = []

    with rasterio.open(str(cog_path)) as src:
        width = src.width
        height = src.height
        crs = src.crs
        nodata = src.nodata

        # Determine bands to read
        read_bands = list(range(1, min(src.count, 3) + 1))

        # Stride over the raster in steps of crop_size
        for row_off in range(0, height - crop_size + 1, crop_size):
            for col_off in range(0, width - crop_size + 1, crop_size):
                window = Window(col_off, row_off, crop_size, crop_size)
                
                # Read window data (C, H, W)
                window_data = src.read(read_bands, window=window)
                
                rgb_uint8, valid_mask = normalize_to_rgb8(window_data, nodata=nodata)
                valid_pixel_frac = float(np.mean(valid_mask))

                # Step 1 Requirement: Skip crops that are more than 50% nodata/black
                if valid_pixel_frac < min_valid_fraction:
                    continue

                # Calculate native bounding box from pixel coords
                # window corners: upper-left and lower-right
                x_min, y_max = src.xy(row_off, col_off, offset="ul")
                x_max, y_min = src.xy(row_off + crop_size, col_off + crop_size, offset="lr")
                
                bounds_native = [
                    float(min(x_min, x_max)),
                    float(min(y_min, y_max)),
                    float(max(x_min, x_max)),
                    float(max(y_min, y_max)),
                ]

                # Reproject to WGS84 (EPSG:4326)
                bounds_wgs84 = bounds_native
                if crs:
                    try:
                        w_bounds = transform_bounds(crs, "EPSG:4326", *bounds_native)
                        bounds_wgs84 = [float(w_bounds[0]), float(w_bounds[1]), float(w_bounds[2]), float(w_bounds[3])]
                    except Exception as ex:
                        logger.warning("Failed to reproject crop bounds to EPSG:4326: %s", ex)

                # Generate unique tile ID
                tile_id = f"{scene_id}_crop_{col_off:05d}_{row_off:05d}"
                crop_file = scene_crops_dir / f"{tile_id}.png"

                # Save as PNG
                img = Image.fromarray(rgb_uint8)
                img.save(crop_file, format="PNG")

                crop_info = CropInfo(
                    tile_id=tile_id,
                    scene_id=scene_id,
                    col_off=col_off,
                    row_off=row_off,
                    file_path=crop_file,
                    bounds_native=bounds_native,
                    bounds_wgs84=bounds_wgs84,
                    valid_pixel_frac=valid_pixel_frac,
                )
                crops_info.append(crop_info)
                if on_progress is not None:
                    on_progress(len(crops_info))

    logger.info("Extracted %d valid crops for scene %s", len(crops_info), scene_id)
    return crops_info
