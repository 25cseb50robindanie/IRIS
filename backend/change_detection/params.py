"""IRIS change detection — named calibration parameters.

Every value here is a reasonable starting point, NOT a validated one (architecture.md, "Calibration &
Validation Strategy"). They are collected in one place so the OSCD calibration and the ablation study can
change them without touching pipeline logic.
"""

# --- Windowed processing ---------------------------------------------------------------------------------
STRIP_ROWS = 512  # full-width strips; keeps peak memory independent of scene height (a 10980-px strip of 4 bands is ~90 MB)

# --- Phase 1: quality masking ----------------------------------------------------------------------------
# SCL classes treated as valid observations (ESA L2A: 4 vegetation, 5 bare soil, 6 water, 7 unclassified,
# 11 snow/ice). Snow is kept but flagged. Everything else (0 no-data, 1 saturated, 2 dark area, 3 cloud shadow,
# 8/9 cloud, 10 cirrus) is invalid.
SCL_VALID_CLASSES = (4, 5, 6, 7, 11)
SCL_SNOW_CLASS = 11

# --- Phase 2: grid check and alignment -------------------------------------------------------------------
MIN_MUTUAL_COVERAGE = 0.30  # below this the pair is "insufficient_evidence" and is not differenced
ECC_RHO_MIN = 0.8  # a converged ECC transform below this correlation is treated as a failure
ECC_MAX_ITERATIONS = 100
ECC_EPSILON = 1e-6
ECC_GAUSS_FILT_SIZE = 5  # passed explicitly: some OpenCV builds reject a None mask positionally
ECC_MAX_PIXELS = 9_000_000  # ECC runs on a decimated copy no larger than this; coarse-to-fine pyramid
ORB_FEATURES = 5000
ORB_MIN_MATCHES = 15
ORB_MIN_INLIER_RATIO = 0.30
ORB_RANSAC_REPROJ_THRESHOLD = 3.0
# Scenes already share a grid when they reach this stage, so residual misregistration is a few pixels at most.
# A transform that moves the image corners further than this is a bad local optimum, not a real alignment.
MAX_ALIGNMENT_SHIFT_PX = 32.0
IDENTITY_SHIFT_PX = 0.02  # below this the warp is skipped so the imagery is not needlessly resampled

# --- Phase 3: radiometric normalisation ------------------------------------------------------------------
PIF_PERCENTILE = 20.0  # bottom 20% of |A - B| across bands = pseudo-invariant pixels
PIF_MIN_PIXELS = 1000
PIF_SLOPE_RANGE = (0.5, 2.0)  # a fitted gain outside this is not a plausible radiometric correction
PIF_SAMPLE_STRIPS = 8  # evenly spaced strips sampled to estimate the percentile threshold
PIF_ITERATIONS = 2  # residual-based re-selections after the selection-free starting fit (see radiometry.py)

# --- Phase 4: change detection ---------------------------------------------------------------------------
KMEANS_SAMPLE_MAX = 2_000_000  # K-means is fitted on at most this many valid pixels; all pixels are then labelled
KMEANS_SEED = 0  # fixed so the same pair always yields the same result (reproducible evaluation)
# The change cluster's centroid must exceed this fraction of the sensor's reflectance range, otherwise the
# pair is reported as "no change": K-means always splits into two clusters, even pure noise.
MIN_CHANGE_MAGNITUDE = 0.03
MORPH_KERNEL_SIZE = 3
MIN_BLOB_PIXELS = 100  # minimum mapping unit: 1 ha at 10 m. Below this, field-boundary noise dominates multi-year pairs
CONNECTIVITY_NOTE = "scipy.ndimage.label default (4-connectivity)"

# --- Merging nearby blobs into one detection ----------------------------------------------------------------
# Blobs whose masks touch after growing by this many pixels (100 m at 10 m) share a detection. The second value is the
# fallback when a group at the first is too big to be one place (see LARGE_AREA_MAX_* below).
NEAR_DILATIONS = (10, 5)
GROUP_DILATION_PX = NEAR_DILATIONS[0]
# Second pass for large-area changes (an airport, a quarry, a township): a wider dilation finds blobs that belong to one
# big change. Such a group becomes ONE detection when it has this many blobs, or covers more than this many pixels;
# smaller groups keep whatever the first pass gave them. Blobs merge when their gap is up to twice the dilation.
LARGE_AREA_DILATIONS = (30, 20)  # widest first (30 px = 300 m at 10 m); a blob takes the widest level that holds
LARGE_AREA_DILATION_PX = LARGE_AREA_DILATIONS[0]
LARGE_AREA_MIN_BLOBS = 3
LARGE_AREA_MIN_PIXELS = 500  # 5 ha at 10 m: strictly more than this merges
# Guard against runaway chaining, applied at EVERY level. Groups merge transitively, so on a scene dense with
# field-scale change (a multi-year pair over farmland) even a 10 px dilation links blobs across the whole tile: on the
# Jewar 2022-2026 pair 28,470 blobs became one 151,937 ha "detection" at 30 px and 12,458 blobs one 94,522 ha one at
# 10 px. A group wider or larger than this is not one change: its blobs fall back to the next narrower level, and
# stay individual candidates if none holds.
LARGE_AREA_MAX_EXTENT_PX = 500  # longest side of a group's bounding box: 5 km at 10 m
LARGE_AREA_MAX_PIXELS = 250_000  # 2,500 ha at 10 m

# --- Phase 5: scoring ------------------------------------------------------------------------------------
CONFIDENCE_WEIGHTS = {
    "alignment_quality": 0.35,
    "cluster_distance": 0.35,
    "terrain_flatness": 0.15,
    "valid_coverage": 0.15,
}
# PLACEHOLDER: no DEM (CartoDEM) is loaded yet, so terrain is treated as flat for every candidate.
# Replace with cos(slope) floored at cos(60 deg) once elevation data exists (architecture.md, Phase 2 step 3).
TERRAIN_FLATNESS_PLACEHOLDER = 1.0
# Two scenes of one MGRS tile share a pixel grid by construction, which is good but not measured. Only a genuine
# ECC correlation above the rho gate earns a higher alignment score than this.
SAME_TILE_ALIGNMENT_QUALITY = 0.90
MIN_CANDIDATE_VALID_COVERAGE = 0.30  # candidates on a thin sliver of valid data are dropped as "Insufficient Evidence"
MIN_STORED_CONFIDENCE = 0.3  # candidates scoring below this are not stored at all
MAX_STORED_CANDIDATES = 5000  # safety cap per pair; the API shows at most DISPLAY_CAP_PER_JOB of them
# When the cap bites, this many of the slots go to the LARGEST detections whatever their confidence, so the cap can
# never drop the biggest changes (the ones an analyst sorts by area to find). The rest go to the most confident.
STORED_AREA_RESERVE = 500
DISPLAY_CAP_PER_JOB = 200  # candidates shown per pair, highest confidence first
NDVI_DROP_SIG = 0.15  # |mean dNDVI| above this labels a blob vegetation gain/loss

# --- Seasonal persistence filter (seasonal.py) -------------------------------------------------------------------
# A vegetation drop can be the calendar (harvest, dry season) rather than an event. The candidate's location is looked up
# in earlier years of the same season; if the current NDVI is within normal variation of what that season usually looks
# like, the drop is seasonal.
SEASONAL_TYPES = ("clearance", "vegetation_loss", "vegetation_retreat")  # vegetation_loss = name used before Phase 4b
SEASONAL_WINDOW_DAYS = 30  # a prior scene counts when it was taken within this many days of the same date in its year
SEASONAL_MIN_PRIORS = 3  # fewer clear priors than this and the check cannot be made: "unverified"
SEASONAL_SIGMAS = 2.0  # a drop within this many std devs of the seasonal history is "seasonal", beyond it "anomalous"
SEASONAL_MIN_STD = 0.05  # floor on the history's std dev: three near-identical scenes must not make every drop anomalous
SEASONAL_MIN_VALID = 0.5  # a prior scene is clear at a candidate when this share of its pixels is valid (SCL, data mask)
SEASONAL_CONFIDENCE_FACTOR = 0.5  # confidence multiplier for a seasonal candidate
UNVERIFIED_CONFIDENCE_FACTOR = 0.85  # ... and for one that could not be checked
SEASONAL_MAX_SAMPLE_PX = 256  # a candidate's window is decimated to at most this many pixels a side when sampled

# --- Ablation (ablation.py) ----------------------------------------------------------------------------------------
MAX_ABLATION_STORED = 5000  # raw detections stored per pair (largest first); the true total is always recorded

# --- Phase 4b: direction classification -------------------------------------------------------------------
# Per-date Classification Components. Index thresholds are strongly scene-dependent (architecture.md prefers
# per-scene Otsu); these fixed values are the domain-standard starting points.
NDVI_VEG_MIN = 0.3  # vegetation: NDVI above this
NDVI_BARE_MAX = 0.15  # bare/built: NDVI below this and not water
NDWI_WATER_MIN = 0.0  # water: (B03 - B08) / (B03 + B08) above this (green and NIR, both 10 m)
# Direction rules
DIRECTION_AREA_GROWTH = 1.2  # same dominant class covering > 1.2x more of the blob = expansion
DIRECTION_AREA_SHRINK = 0.8  # ... < 0.8x = contraction
# |mean dNDVI| needed to call NDVI "dropped" or "increased" when naming a change. Smaller than NDVI_DROP_SIG on
# purpose: bare ground already sits below NDVI 0.15, so it can never drop by 0.15.
NDVI_DIRECTION_MIN = 0.05
# SWIR-free stand-in for "bare ground with higher NDBI" (no SWIR band is ingested): visible brightness up by this
# fraction. A proxy to be calibrated, not a measurement.
BUILT_BRIGHTNESS_GAIN = 0.15
# Query words that hint at a direction add this to a matching candidate's combined search score
DIRECTION_SEARCH_BOOST = 0.2
