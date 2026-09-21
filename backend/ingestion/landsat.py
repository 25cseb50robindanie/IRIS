"""IRIS Ingestion — Landsat 8/9 Collection 2 Level-2 (L2SP) reader and COG builder.

A Landsat scene is a folder of single-band GeoTIFFs sharing one 30 m grid:

    LC08_L2SP_146040_20240115_20240125_02_T1_SR_B2.TIF  blue      \\
    ..._SR_B3.TIF green, ..._SR_B4.TIF red, ..._SR_B5.TIF NIR      } surface reflectance, uint16
    ..._QA_PIXEL.TIF                                                 bit-packed quality flags, uint16

It yields the same three rasters per scene as a Sentinel-2 product, on the same conventions, so change detection needs
no sensor-specific branches:
  * display COG   - B4/B3/B2 as 8-bit RGB with the fixed reflectance stretch Sentinel-2 uses
  * analysis COG  - B2/B3/B4/B5 (blue, green, red, NIR) as uint16 in the Sentinel-2 digital-number convention:
                    reflectance = (DN - 1000) / 10000. Landsat's own scaling (reflectance = DN * 2.75e-5 - 0.2) is
                    applied here once, so NDVI, NDWI and the reflectance thresholds mean the same for both sensors.
  * mask COG      - QA_PIXEL decoded into Sentinel-2 SCL class codes (0 no data, 9 cloud, 3 cloud shadow, 11 snow,
                    7 clear), so the existing validity logic reads it unchanged. Stored as the scene's `scl_path`.

QA_PIXEL bits used: 0 fill, 3 cloud, 4 cloud shadow, 5 snow. Invalid = cloud OR shadow; snow is kept but flagged.
(Bits 1 dilated cloud and 2 cirrus are not used: the brief names 3, 4 and 5.)
"""

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
from rasterio.windows import Window

from ingestion.loader import SceneMetadata, compute_sha256, io_path, plain_path, read_raster_bounds
from ingestion.safe import STRIP_ROWS, Sentinel2Outputs, _display_rgb

logger = logging.getLogger("iris.ingestion.landsat")

# LC08_L2SP_<path><row>_<acquired>_<processed>_<collection>_<tier>
LANDSAT_NAME = re.compile(
    r"^(?P<mission>LC0[89])_L2SP_(?P<path>\d{3})(?P<row>\d{3})_(?P<acquired>\d{8})_(?P<processed>\d{8})_(?P<collection>\d{2})_(?P<tier>T1|T2|RT)",
    re.IGNORECASE,
)
BANDS = ("B2", "B3", "B4", "B5")  # blue, green, red, NIR: the order of the analysis COG bands
_BAND_SUFFIX = {"B2": "SR_B2", "B3": "SR_B3", "B4": "SR_B4", "B5": "SR_B5"}
QA_SUFFIX = "QA_PIXEL"
MAX_SEARCH_DEPTH = 1  # a scene folder may itself contain one folder of bands

# Collection 2 Level-2 surface reflectance scaling, and the Sentinel-2 convention it is converted to
LANDSAT_SCALE, LANDSAT_OFFSET = 2.75e-5, -0.2
S2_QUANTIFICATION, S2_ADD_OFFSET = 10000.0, -1000.0

# Sentinel-2 SCL codes the QA_PIXEL bits are written as
SCL_NODATA, SCL_SHADOW, SCL_CLEAR, SCL_CLOUD, SCL_SNOW = 0, 3, 7, 9, 11

_SENSOR = {"LC08": "landsat8", "LC09": "landsat9"}


@dataclass
class LandsatProduct:
    """A validated Landsat Collection 2 Level-2 scene on disk."""
    root: Path
    scene_id: str
    sensor: str  # landsat8 | landsat9
    acquisition_date: str
    wrs_path_row: str
    band_paths: Dict[str, Path]  # B2..B5 surface reflectance
    qa_path: Path
    add_offsets: Dict[str, float] = field(default_factory=dict)


def parse_landsat_name(name: str) -> Optional[re.Match]:
    return LANDSAT_NAME.match(name)


def _files_for(directory: Path, prefix: str, suffix: str) -> list:
    """Files `<prefix>_<suffix>.TIF` in a directory (any case of the extension)."""
    want = f"{prefix}_{suffix}".upper()
    return [p for p in directory.iterdir() if p.is_file() and p.suffix.upper() in (".TIF", ".TIFF") and p.stem.upper() == want]


def open_landsat_product(path: Path) -> Optional[LandsatProduct]:
    """Validate a Landsat C2 L2 scene given as a folder or as one of its files.

    Returns None when `path` is not a Landsat L2SP scene (the caller tries another format); raises ValueError for a scene
    that is present but unusable, with a message safe to show the analyst.
    """
    base = io_path(path.resolve())
    if base.is_file():
        match = parse_landsat_name(base.name)
        if match is None:
            return None
        folder, prefix = base.parent, base.name[: match.end()]
    elif base.is_dir():
        folder, prefix = None, None
        candidates = [(base, 0)]
        found: Dict[str, Path] = {}  # prefix -> folder
        while candidates:
            directory, depth = candidates.pop()
            try:
                children = sorted(directory.iterdir())
            except OSError:
                continue
            for child in children:
                if child.is_file():
                    m = parse_landsat_name(child.name)
                    if m and child.suffix.upper() in (".TIF", ".TIFF"):
                        found.setdefault(child.name[: m.end()], directory)
                elif child.is_dir() and not child.is_symlink() and depth < MAX_SEARCH_DEPTH:
                    candidates.append((child, depth + 1))
        if not found:
            return None
        if len(found) > 1:
            raise ValueError("More than one Landsat scene found; select a single scene folder")
        prefix, folder = next(iter(found.items()))
    else:
        return None

    match = parse_landsat_name(prefix)
    assert match is not None
    root_io = folder.resolve()

    def pick(suffix: str) -> Path:
        matches = _files_for(folder, prefix, suffix)
        if len(matches) != 1:
            raise ValueError(f"Could not find exactly one {suffix} file for {prefix}")
        resolved = matches[0].resolve()
        if not resolved.is_relative_to(root_io):
            raise ValueError(f"{suffix} resolves outside the scene folder")
        return plain_path(resolved)

    band_paths = {b: pick(_BAND_SUFFIX[b]) for b in BANDS}
    qa_path = pick(QA_SUFFIX)
    acquired = match.group("acquired")
    return LandsatProduct(
        root=plain_path(folder),
        scene_id=re.sub(r"[^\w\-_.]", "_", prefix.upper()),
        sensor=_SENSOR[match.group("mission").upper()],
        acquisition_date=f"{acquired[:4]}-{acquired[4:6]}-{acquired[6:]}",
        wrs_path_row=f"{match.group('path')}/{match.group('row')}",
        band_paths=band_paths,
        qa_path=qa_path,
        add_offsets={b: S2_ADD_OFFSET for b in BANDS},
    )


def compute_landsat_checksum(product: LandsatProduct) -> str:
    """One SHA-256 over every source raster we read, for the provenance record."""
    parts = {p.name: compute_sha256(p) for p in (*product.band_paths.values(), product.qa_path)}
    digest = hashlib.sha256()
    for name in sorted(parts):
        digest.update(f"{name}:{parts[name]}\n".encode("utf-8"))
    return digest.hexdigest()


def inspect_landsat_product(product: LandsatProduct) -> SceneMetadata:
    """Capability detection: CRS, grid and footprint come from the 30 m red band."""
    with rasterio.open(str(product.band_paths["B4"])) as src:
        crs = src.crs.to_string() if src.crs else "unknown"
        res = (float(src.res[0]), float(src.res[1]))
        width, height = src.width, src.height
    bounds, bounds_wgs84 = read_raster_bounds(product.band_paths["B4"])
    metadata = SceneMetadata(
        scene_id=product.scene_id,
        sensor=product.sensor,
        acquisition_date=product.acquisition_date,
        crs=crs,
        driver="Landsat-C2-L2",
        bands=len(BANDS),
        dtype="uint16",
        nodata=0.0,
        bounds=bounds,
        bounds_wgs84=bounds_wgs84,
        resolution=res,
        width=width,
        height=height,
        raw_checksum=compute_landsat_checksum(product),
    )
    logger.info(
        "Capability detection: Landsat C2 L2 %s scene_id=%s path/row=%s date=%s crs=%s size=%dx%d res=%s",
        product.sensor, product.scene_id, product.wrs_path_row, product.acquisition_date, crs, width, height, res,
    )
    return metadata


def decode_qa(qa: np.ndarray) -> Dict[str, np.ndarray]:
    """QA_PIXEL (uint16, bit-packed) -> boolean masks. cloud = bit 3, shadow = bit 4, snow = bit 5, fill = bit 0."""
    qa = qa.astype(np.uint16, copy=False)
    return {
        "fill": (qa & 1).astype(bool),
        "cloud": ((qa >> 3) & 1).astype(bool),
        "shadow": ((qa >> 4) & 1).astype(bool),
        "snow": ((qa >> 5) & 1).astype(bool),
    }


def qa_to_scl(qa: np.ndarray, no_data: np.ndarray) -> np.ndarray:
    """QA_PIXEL -> Sentinel-2 SCL class codes (uint8). Cloud and shadow win over snow; no data wins over everything."""
    flags = decode_qa(qa)
    scl = np.full(qa.shape, SCL_CLEAR, dtype=np.uint8)
    scl[flags["snow"]] = SCL_SNOW
    scl[flags["shadow"]] = SCL_SHADOW
    scl[flags["cloud"]] = SCL_CLOUD
    scl[flags["fill"] | no_data] = SCL_NODATA
    return scl


def to_s2_dn(raw: np.ndarray) -> np.ndarray:
    """Landsat C2 L2 surface-reflectance DN -> the Sentinel-2 DN convention (reflectance = (DN - 1000) / 10000).

    A raw DN of 0 is fill and stays 0 (no data); every other value is floored at 1 so a valid dark pixel is not no data.
    """
    reflectance = raw.astype(np.float32) * LANDSAT_SCALE + LANDSAT_OFFSET
    converted = np.clip(np.rint(reflectance * S2_QUANTIFICATION - S2_ADD_OFFSET), 1, 65535)
    return np.where(raw == 0, 0, converted).astype(np.uint16)


def build_landsat_cogs(
    product: LandsatProduct,
    output_dir: Path = Path("data/cogs"),
    staging_dir: Path = Path("data/staging"),
    on_stage: Optional[Callable[[str, int, int], None]] = None,
) -> Sentinel2Outputs:
    """Write the analysis, mask and display COGs for a scene and compute its cloud percentage.

    Same passes and stage names as the Sentinel-2 builder except that the mask pass is called "qa":
    "bands", "qa", "rgb", "cog". Every file is written under staging/ and moved into place with an atomic rename.
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

    offsets = [S2_ADD_OFFSET] * len(BANDS)
    data_px = cloud_px = snow_px = total_px = 0
    sources = {b: rasterio.open(str(p)) for b, p in product.band_paths.items()}
    qa_src = rasterio.open(str(product.qa_path))
    try:
        ref = sources["B4"]
        width, height = ref.width, ref.height
        for name, src in {**sources, "QA_PIXEL": qa_src}.items():
            if (src.crs, src.transform, src.width, src.height) != (ref.crs, ref.transform, width, height):
                raise ValueError(f"{name} is not on the same grid as B4")
        base = {
            "driver": "GTiff", "width": width, "height": height, "crs": ref.crs, "transform": ref.transform,
            "tiled": True, "blockxsize": 512, "blockysize": 512, "compress": "DEFLATE", "BIGTIFF": "IF_SAFER",
        }

        # ---- bands (reflectance -> Sentinel-2 DN) and qa (bit-packed flags -> SCL codes), one strip at a time -------
        t0 = time.monotonic()
        report("bands", 0, height)
        report("qa", 0, height)
        with rasterio.open(
            str(targets["analysis"][0]), "w", count=4, dtype="uint16", nodata=0, predictor=2, **base
        ) as analysis, rasterio.open(str(targets["scl"][0]), "w", count=1, dtype="uint8", nodata=0, **base) as mask:
            for row in range(0, height, STRIP_ROWS):
                rows = min(STRIP_ROWS, height - row)
                window = Window(0, row, width, rows)
                raw = np.stack([sources[b].read(1, window=window) for b in BANDS])
                analysis.write(np.stack([to_s2_dn(r) for r in raw]), window=window)
                scl = qa_to_scl(qa_src.read(1, window=window), no_data=(raw == 0).all(axis=0))
                mask.write(scl, 1, window=window)

                valid_data = scl != SCL_NODATA
                total_px += scl.size
                data_px += int(valid_data.sum())
                cloud_px += int(((scl == SCL_CLOUD) | (scl == SCL_SHADOW)).sum())
                snow_px += int((scl == SCL_SNOW).sum())
                report("bands", row + rows, height)
                report("qa", row + rows, height)
            analysis.update_tags(
                BANDS="B2,B3,B4,B5", PRODUCT=scene_id, BOA_ADD_OFFSETS=",".join(str(o) for o in offsets),
                SOURCE_SCALING="landsat_c2_l2_sr: reflectance = DN * 2.75e-5 - 0.2, rewritten as (DN' - 1000) / 10000",
            )
            for idx, band in enumerate(BANDS, start=1):
                analysis.set_band_description(idx, band)
            mask.update_tags(SOURCE="QA_PIXEL", CODES="0 fill, 3 shadow, 7 clear, 9 cloud, 11 snow")
        timings["bands_qa_s"] = round(time.monotonic() - t0, 2)
    finally:
        for src in sources.values():
            src.close()
        qa_src.close()

    # ---- rgb: display COG from the analysis COG (B4/B3/B2 -> R/G/B) ------------------------------------------------
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

    # ---- cog: overviews (nearest for the categorical mask), then atomic rename -----------------------------------
    from rasterio.enums import Resampling

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

    cloud_pct = round(100.0 * cloud_px / data_px, 2) if data_px else 0.0
    snow_pct = round(100.0 * snow_px / data_px, 2) if data_px else 0.0
    nodata_pct = round(100.0 * (total_px - data_px) / total_px, 2) if total_px else 0.0
    timings["total_s"] = round(time.monotonic() - t_start, 2)
    logger.info(
        "Landsat COGs written for %s: cloud=%.2f%% (incl. shadow) snow=%.2f%% nodata=%.2f%% in %.1fs",
        scene_id, cloud_pct, snow_pct, nodata_pct, time.monotonic() - t_start,
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
