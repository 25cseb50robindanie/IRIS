"""IRIS Ingestion — Sentinel-2 L2A .SAFE product reader and COG builder.

A .SAFE product yields three rasters per scene, all on the same 10 m grid:
  * display COG   — B04/B03/B02 as 8-bit RGB (what titiler serves to the map)
  * analysis COG  — B02/B03/B04/B08 as uint16 digital numbers (what change detection differences)
  * SCL COG       — the Scene Classification Layer, resampled 20 m -> 10 m with nearest-neighbour

Everything is processed in full-width strips so a 10980 x 10980 tile never has to fit in memory.
"""

import hashlib
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from ingestion.loader import (
    SceneMetadata,
    compute_sha256,
    detect_acquisition_date,
    io_path,
    plain_path,
    read_raster_bounds,
)

logger = logging.getLogger("iris.ingestion.safe")

MTD_L2A = "MTD_MSIL2A.xml"
MTD_L1C = "MTD_MSIL1C.xml"
L2A_PRODUCT_TYPES = ("S2MSI2A",)
L2A_PROCESSING_LEVELS = ("Level-2A",)
MAX_MTD_BYTES = 20 * 1024 * 1024  # a real MTD is well under 1 MB
MAX_SEARCH_DEPTH = 3  # how far below the selected folder to look for the product
MAX_DIRS_SCANNED = 2000

BANDS_10M = ("B02", "B03", "B04", "B08")  # blue, green, red, NIR — the order of the analysis COG bands
_BAND_ID = {"B02": 1, "B03": 2, "B04": 3, "B08": 7}  # band_id used by BOA_ADD_OFFSET in the product metadata
_BAND_FILE = re.compile(r"^[A-Za-z0-9_]+_(B02|B03|B04|B08)_10m\.jp2$")
_SCL_FILE = re.compile(r"^[A-Za-z0-9_]+_SCL_20m\.jp2$")

STRIP_ROWS = 1024  # even, and a multiple of the 512 COG block size
QUANTIFICATION = 10000.0  # L2A digital number -> reflectance
DISPLAY_REFLECTANCE_MAX = 0.30  # fixed stretch, identical for every scene so before/after views compare fairly

# Scene Classification Layer classes (ESA L2A). Class 6 is WATER and class 11 is snow/ice.
SCL_NODATA = 0
SCL_CLOUD_CLASSES = (3, 8, 9, 10)  # cloud shadow, cloud (medium/high probability), thin cirrus
SCL_SNOW = 11
SCL_CLASS_COUNT = 12


@dataclass
class Sentinel2Product:
    """A validated Sentinel-2 L2A product on disk."""
    root: Path
    scene_id: str
    acquisition_date: str
    processing_baseline: Optional[str]
    band_paths: Dict[str, Path]  # B02/B03/B04/B08 at 10 m
    scl_path: Path  # SCL at 20 m
    mtd_path: Path
    add_offsets: Dict[str, float] = field(default_factory=dict)  # BOA_ADD_OFFSET per band (DN units)


@dataclass
class Sentinel2Outputs:
    """What build_sentinel2_cogs produced."""
    display_cog: Path
    analysis_cog: Path
    scl_cog: Path
    cloud_pct: float  # % of data pixels that are cloud, cloud shadow or cirrus
    snow_pct: float  # % of data pixels that are snow/ice (flagged, not treated as invalid)
    nodata_pct: float  # % of the tile with no data
    width: int
    height: int
    timings: Dict[str, float] = field(default_factory=dict)


def find_safe_root(path: Path) -> Optional[Path]:
    """Locate the folder holding MTD_MSIL2A.xml at or below `path` (a .SAFE folder or one that contains it).

    Returns None when no Sentinel-2 L2A product is present. Symlinked directories are not followed, so a
    link inside the selected folder cannot lead the scan somewhere the analyst did not choose.
    """
    base = io_path(path.resolve())
    if not base.is_dir():
        return None

    found: List[Path] = []
    saw_l1c = False
    scanned = 0

    def scan(directory: Path, depth: int) -> None:
        nonlocal saw_l1c, scanned
        scanned += 1
        if scanned > MAX_DIRS_SCANNED:
            return
        if (directory / MTD_L2A).is_file():
            found.append(directory)
            return
        if (directory / MTD_L1C).is_file():
            saw_l1c = True
            return
        if depth >= MAX_SEARCH_DEPTH:
            return
        try:
            children = sorted(directory.iterdir())
        except OSError:
            return
        for child in children:
            if child.is_dir() and not child.is_symlink():
                scan(child, depth + 1)

    scan(base, 0)

    if len(found) > 1:
        raise ValueError("More than one Sentinel-2 product found; select a single .SAFE folder")
    if not found:
        if saw_l1c:
            raise ValueError("This is a Sentinel-2 Level-1C product; only Level-2A (MTD_MSIL2A.xml) is supported")
        return None
    return plain_path(found[0])


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_mtd(mtd_path: Path) -> Dict[str, object]:
    """Read the handful of fields we need from MTD_MSIL2A.xml (namespace-agnostic)."""
    mtd_io = io_path(mtd_path)
    if mtd_io.stat().st_size > MAX_MTD_BYTES:
        raise ValueError("MTD_MSIL2A.xml is unexpectedly large; refusing to parse it")
    try:
        tree = ET.parse(str(mtd_io))
    except ET.ParseError as e:
        raise ValueError(f"MTD_MSIL2A.xml is not valid XML: {e}") from e

    info: Dict[str, object] = {"offsets_by_id": {}}
    for el in tree.getroot().iter():
        name = _local(el.tag)
        text = (el.text or "").strip()
        if name == "PRODUCT_TYPE":
            info["product_type"] = text
        elif name == "PROCESSING_LEVEL":
            info["processing_level"] = text
        elif name == "PRODUCT_URI":
            info["product_uri"] = text
        elif name == "PRODUCT_START_TIME":
            info["start_time"] = text
        elif name == "PROCESSING_BASELINE":
            info["baseline"] = text
        elif name == "BOA_ADD_OFFSET" and text:
            try:
                info["offsets_by_id"][int(el.attrib.get("band_id", "-1"))] = float(text)  # type: ignore[index]
            except ValueError:
                pass
    return info


def open_safe_product(path: Path) -> Optional[Sentinel2Product]:
    """Validate a Sentinel-2 L2A product and locate the bands change detection needs.

    Returns None if `path` holds no L2A product (caller decides the fallback); raises ValueError for a
    product that is present but unusable, with a message safe to show the analyst.
    """
    root = find_safe_root(path)
    if root is None:
        return None

    mtd_path = root / MTD_L2A
    info = _parse_mtd(mtd_path)
    product_type = str(info.get("product_type", ""))
    level = str(info.get("processing_level", ""))
    if (product_type or level) and product_type not in L2A_PRODUCT_TYPES and level not in L2A_PROCESSING_LEVELS:
        raise ValueError(f"Not a Level-2A product (PRODUCT_TYPE={product_type or '?'}, PROCESSING_LEVEL={level or '?'})")

    root_io = io_path(root)
    granule_root = root_io / "GRANULE"
    granules = [d for d in granule_root.iterdir() if d.is_dir()] if granule_root.is_dir() else []
    if len(granules) != 1:
        raise ValueError(f"Expected exactly one granule under GRANULE/, found {len(granules)}")
    img_data = granules[0] / "IMG_DATA"
    r10m, r20m = img_data / "R10m", img_data / "R20m"

    def pick(directory: Path, suffix: str, pattern: "re.Pattern[str]") -> Path:
        matches = [p for p in directory.glob(f"*_{suffix}*.jp2") if pattern.match(p.name)] if directory.is_dir() else []
        if len(matches) != 1:
            raise ValueError(f"Could not find exactly one {suffix} band in {directory.parent.name}/{directory.name}")
        resolved = matches[0].resolve()
        if not resolved.is_relative_to(root_io):
            raise ValueError(f"{suffix} band resolves outside the product folder")
        return plain_path(resolved)

    band_paths = {b: pick(r10m, f"{b}_10m", _BAND_FILE) for b in BANDS_10M}
    scl_path = pick(r20m, "SCL_20m", _SCL_FILE)

    offsets_by_id: Dict[int, float] = info["offsets_by_id"]  # type: ignore[assignment]
    add_offsets = {b: offsets_by_id.get(_BAND_ID[b], 0.0) for b in BANDS_10M}

    uri = str(info.get("product_uri", "")) or root.name
    scene_id = re.sub(r"[^\w\-_.]", "_", re.sub(r"\.SAFE$", "", uri, flags=re.IGNORECASE))

    start = str(info.get("start_time", ""))
    acquisition_date = start[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", start) else detect_acquisition_date(scene_id)

    return Sentinel2Product(
        root=root,
        scene_id=scene_id,
        acquisition_date=acquisition_date,
        processing_baseline=str(info["baseline"]) if info.get("baseline") else None,
        band_paths=band_paths,
        scl_path=scl_path,
        mtd_path=mtd_path,
        add_offsets=add_offsets,
    )


def compute_safe_checksum(product: Sentinel2Product) -> str:
    """One SHA-256 over the metadata file and every source raster we read, for the provenance record."""
    parts = {product.mtd_path.name: compute_sha256(product.mtd_path), product.scl_path.name: compute_sha256(product.scl_path)}
    for path in product.band_paths.values():
        parts[path.name] = compute_sha256(path)
    digest = hashlib.sha256()
    for name in sorted(parts):
        digest.update(f"{name}:{parts[name]}\n".encode("utf-8"))
    return digest.hexdigest()


def inspect_safe_product(product: Sentinel2Product) -> SceneMetadata:
    """Capability detection for a product: CRS, grid and footprint come from the 10 m red band."""
    with rasterio.open(str(product.band_paths["B04"])) as src:
        crs = src.crs.to_string() if src.crs else "unknown"
        res = (float(src.res[0]), float(src.res[1]))
        width, height = src.width, src.height
    bounds, bounds_wgs84 = read_raster_bounds(product.band_paths["B04"])

    metadata = SceneMetadata(
        scene_id=product.scene_id,
        sensor="sentinel2",
        acquisition_date=product.acquisition_date,
        crs=crs,
        driver="SAFE",
        bands=len(BANDS_10M),
        dtype="uint16",
        nodata=0.0,
        bounds=bounds,
        bounds_wgs84=bounds_wgs84,
        resolution=res,
        width=width,
        height=height,
        raw_checksum=compute_safe_checksum(product),
    )
    logger.info(
        "Capability detection: Sentinel-2 L2A scene_id=%s baseline=%s date=%s crs=%s size=%dx%d bands=%s",
        metadata.scene_id,
        product.processing_baseline,
        metadata.acquisition_date,
        metadata.crs,
        width,
        height,
        ",".join(BANDS_10M),
    )
    return metadata


def _display_rgb(stack: np.ndarray, offsets: List[float]) -> np.ndarray:
    """(4,h,w) uint16 DN [B02,B03,B04,B08] -> (3,h,w) uint8 RGB with a fixed reflectance stretch.

    0 is reserved for no-data, so valid dark pixels are floored at 1.
    """
    _, h, w = stack.shape
    rgb = np.empty((3, h, w), dtype=np.uint8)
    for out_i, band_i in enumerate((2, 1, 0)):  # R=B04, G=B03, B=B02
        reflectance = (stack[band_i].astype(np.float32) + offsets[band_i]) / QUANTIFICATION
        rgb[out_i] = np.clip(reflectance / DISPLAY_REFLECTANCE_MAX * 255.0, 1, 255).astype(np.uint8)
    rgb[:, (stack[:3] == 0).all(axis=0)] = 0
    return rgb


def build_sentinel2_cogs(
    product: Sentinel2Product,
    output_dir: Path = Path("data/cogs"),
    staging_dir: Path = Path("data/staging"),
    on_stage: Optional[Callable[[str, int, int], None]] = None,
) -> Sentinel2Outputs:
    """Write the analysis, SCL and display COGs for a product and compute its cloud percentage.

    Four sequential passes, each reported through on_stage(stage, done, total) with stage one of "bands",
    "scl", "rgb", "cog":
      bands - read B02/B03/B04/B08 from the JPEG2000s and write the 4-band analysis COG
      scl   - read the 20 m SCL, resample it by 2x nearest-neighbour (it is categorical) and write it; count classes
      rgb   - build the 8-bit RGB display COG from the analysis COG just written (cheaper than decoding JP2 again)
      cog   - build overviews and move every file into place

    Each file is written under staging/ and moved into place with an atomic rename only once complete,
    so a crash never leaves a half-written COG that looks valid.
    """

    def report(stage: str, done: int, total: int) -> None:
        if on_stage is not None:
            on_stage(stage, done, total)

    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    scene_id = product.scene_id
    timings: Dict[str, float] = {}
    t_start = time.monotonic()

    targets = {
        "display": (staging_dir / f"{scene_id}.tif", output_dir / f"{scene_id}.tif"),
        "analysis": (staging_dir / f"{scene_id}_analysis.tif", output_dir / f"{scene_id}_analysis.tif"),
        "scl": (staging_dir / f"{scene_id}_scl.tif", output_dir / f"{scene_id}_scl.tif"),
    }
    for staged, _ in targets.values():
        staged.unlink(missing_ok=True)

    scl_counts = np.zeros(SCL_CLASS_COUNT + 1, dtype=np.int64)
    offsets = [product.add_offsets.get(b, 0.0) for b in BANDS_10M]

    sources = {b: rasterio.open(str(p)) for b, p in product.band_paths.items()}
    scl_src = rasterio.open(str(product.scl_path))
    try:
        ref = sources["B04"]
        width, height = ref.width, ref.height
        for band, src in sources.items():
            if (src.crs, src.transform, src.width, src.height) != (ref.crs, ref.transform, width, height):
                raise ValueError(f"{band} is not on the same 10 m grid as B04")
        if scl_src.width * 2 != width or scl_src.height * 2 != height:
            raise ValueError("SCL is not exactly half the 10 m resolution; cannot resample it by 2x nearest-neighbour")

        base = {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "crs": ref.crs,
            "transform": ref.transform,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "DEFLATE",
            "BIGTIFF": "IF_SAFER",
        }

        # ---- bands: analysis COG -------------------------------------------------------------------------------
        t0 = time.monotonic()
        report("bands", 0, height)
        with rasterio.open(
            str(targets["analysis"][0]), "w", count=4, dtype="uint16", nodata=0, predictor=2, **base
        ) as analysis:
            for row in range(0, height, STRIP_ROWS):
                rows = min(STRIP_ROWS, height - row)
                window = Window(0, row, width, rows)
                analysis.write(np.stack([sources[b].read(1, window=window) for b in BANDS_10M]), window=window)
                report("bands", row + rows, height)
            analysis.update_tags(
                BANDS="B02,B03,B04,B08", PRODUCT=scene_id, BOA_ADD_OFFSETS=",".join(str(o) for o in offsets)
            )
            for idx, band in enumerate(BANDS_10M, start=1):
                analysis.set_band_description(idx, band)
        timings["bands_s"] = round(time.monotonic() - t0, 2)

        # ---- scl: categorical, so nearest-neighbour only, never bilinear --------------------------------------
        t0 = time.monotonic()
        report("scl", 0, height)
        with rasterio.open(str(targets["scl"][0]), "w", count=1, dtype="uint8", nodata=0, **base) as scl_dst:
            for row in range(0, height, STRIP_ROWS):
                rows = min(STRIP_ROWS, height - row)
                window = Window(0, row, width, rows)
                scl = scl_src.read(
                    1,
                    window=Window(0, row // 2, scl_src.width, rows // 2),
                    out_shape=(rows, width),
                    resampling=Resampling.nearest,
                )
                scl_dst.write(scl, 1, window=window)
                scl_counts += np.bincount(scl.ravel(), minlength=SCL_CLASS_COUNT + 1)[: SCL_CLASS_COUNT + 1]
                report("scl", row + rows, height)
        timings["scl_s"] = round(time.monotonic() - t0, 2)
    finally:
        for src in sources.values():
            src.close()
        scl_src.close()

    # ---- rgb: display COG from the analysis COG ------------------------------------------------------------------
    t0 = time.monotonic()
    report("rgb", 0, height)
    with rasterio.open(str(targets["analysis"][0])) as analysis_in, rasterio.open(
        str(targets["display"][0]), "w", count=3, dtype="uint8", nodata=0, **base
    ) as display:
        for row in range(0, height, STRIP_ROWS):
            rows = min(STRIP_ROWS, height - row)
            window = Window(0, row, width, rows)
            display.write(_display_rgb(analysis_in.read(window=window), offsets), window=window)
            report("rgb", row + rows, height)
    timings["rgb_s"] = round(time.monotonic() - t0, 2)

    # ---- cog: overviews (averaged for imagery, nearest for the categorical SCL), then atomic rename ------------
    t0 = time.monotonic()
    report("cog", 0, 3)
    for n, (key, resampling) in enumerate(
        (("display", Resampling.average), ("analysis", Resampling.average), ("scl", Resampling.nearest)), start=1
    ):
        with rasterio.open(str(targets[key][0]), "r+") as dst:
            dst.build_overviews([2, 4, 8, 16], resampling)
            dst.update_tags(ns="rio_overview", resampling=resampling.name)
        report("cog", n, 3)
    timings["overviews_s"] = round(time.monotonic() - t0, 2)

    for staged, final in targets.values():
        final.unlink(missing_ok=True)
        os.replace(staged, final)

    total = int(scl_counts.sum())
    data_px = total - int(scl_counts[SCL_NODATA])
    cloud_px = int(sum(scl_counts[c] for c in SCL_CLOUD_CLASSES))
    cloud_pct = round(100.0 * cloud_px / data_px, 2) if data_px > 0 else 0.0
    snow_pct = round(100.0 * int(scl_counts[SCL_SNOW]) / data_px, 2) if data_px > 0 else 0.0
    nodata_pct = round(100.0 * int(scl_counts[SCL_NODATA]) / total, 2) if total > 0 else 0.0
    timings["total_s"] = round(time.monotonic() - t_start, 2)

    logger.info(
        "Sentinel-2 COGs written for %s: cloud=%.2f%% (incl. shadow/cirrus) snow=%.2f%% nodata=%.2f%% in %.1fs",
        scene_id,
        cloud_pct,
        snow_pct,
        nodata_pct,
        time.monotonic() - t_start,
    )
    return Sentinel2Outputs(
        display_cog=targets["display"][1],
        analysis_cog=targets["analysis"][1],
        scl_cog=targets["scl"][1],
        cloud_pct=cloud_pct,
        snow_pct=snow_pct,
        nodata_pct=nodata_pct,
        width=width,
        height=height,
        timings=timings,
    )
