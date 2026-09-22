# IRIS: Architecture Note & Systems Design Specification
**Problem Statement ID: 26227** — *Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery*  
**Proponent:** Ministry of Defence / Indian Army (DGIS)  
**System Designation:** IRIS (Intelligent Retrieval & Interpretation System)  
**Architecture Classification:** Frozen Production Architecture v2.2.7

---

## 1. Architecture Note: Executive Summary & System Requirements

### 1.1 Architectural Context & Purpose
This **Architecture Note** provides the formal system design, operational rationale, component decomposition, and architectural requirements for **IRIS**, a zero-cloud, 100% offline desktop platform designed to solve SIH Problem Statement 26227.

Earth-observation archives are expanding rapidly with multi-temporal, multi-spectral, and multi-sensor imagery. While conventional geospatial catalogues are effective for searching by rigid metadata (coordinates, dates, sensor IDs), intelligence analysts require the ability to retrieve imagery **by semantic meaning** (natural language and image-to-image queries) and to detect **genuine physical changes** over time while suppressing the severe false-alarm rates typical of naive image differencing.

### 1.2 Core Architectural Requirements (Non-Negotiable)
1. **Air-Gapped Operational Sovereignty (100% Offline):** No external cloud calls, API keys, telemetry, or remote dependencies. The entire stack (frontend, tile rendering, vector search, foundation models, raster math, change detection) operates strictly on `localhost` (127.0.0.1).
2. **Explainable Non-Black-Box Change Engine:** Deep-learning end-to-end change detection models are prone to hallucinated changes, lack explainability, and fail defence audit standards. IRIS adopts an interpretable 5-phase pipeline combining physical masking, sub-pixel co-registration, radiometric normalization, block-PCA/K-Means difference clustering, and multi-spectral index directional classification (appearance, disappearance, expansion, contraction).
3. **Bounded-Memory Streaming Processing:** Field workstations have finite RAM (8–16 GB). Raw satellite scenes (e.g., 100 km × 100 km Sentinel-2 tiles) cannot be loaded uncompressed into memory. All raster processing is strictly windowed and streamed via GDAL/Rasterio with 16–32px overlapping halos to eliminate seam artifacts while maintaining bounded memory consumption.
4. **Dynamic Incremental Ingestion & Zero-Downtime Indexing:** New acquisitions must be indexed dynamically without locking the system or requiring full index re-computations. Vector embeddings are appended incrementally to a FAISS HNSW graph, paired with SQLite WAL (Write-Ahead Logging) metadata storage.
5. **Crash Resilience & Atomic Staging:** Desktop tools in field conditions may be terminated mid-computation. IRIS enforces an atomic staging pattern: outputs are written to staging directories and atomically renamed on transaction commit. Interrupted jobs are recovered on startup via an automated reconciliation sweep.

---

## Project Context
Built for Smart India Hackathon-style Problem Statement 26227, issued by the Ministry of Defence / Indian Army (DGIS).
* **The Problem:** Earth-observation archives are searchable by metadata (coordinates, date, sensor) but not by semantic meaning or by change over time. Existing change-detection tools produce excessive false alarms from clouds, seasonal variations, illumination shifts, and geometric registration errors.
* **The Goal:** Build an offline system that makes an imagery archive queryable by meaning and by change, suppressing false alarms through an explainable, non-black-box decision pipeline.

---

## Non-Negotiable Constraints
1. **Fully Offline:** No cloud services, external APIs, or network calls except to `localhost`.
2. **Single-Analyst Local Tool:** No login or authentication required.
3. **Explainable Deliverables:** Source code and architecture notes must be transparent; decision-making stages remain interpretable and reproducible.
4. **Open-Access Data Only:** Restricted to publicly accessible datasets under applicable licences (Sentinel-2, Sentinel-1, Landsat Collection 2, Bhuvan/NRSC open products).

---

## Tech Stack
* **App Shell:** Electron (Chromium-based desktop wrapper, reuses web frontend skills).
* **Frontend:** React + TailwindCSS + shadcn/ui (QGIS-inspired light/neutral visual theme).
* **Map Rendering:** MapLibre GL — open-source, offline raster tile source, no internet basemap/OSM overlay.
* **Backend:** Python + FastAPI.
* **Raster I/O:** `rasterio` / GDAL, implemented with chunked, window-by-window processing rather than loading entire scenes into memory — keeps memory use bounded regardless of scene size, on modest field hardware.
* **Tile Serving:** `titiler` — serves COGs as XYZ map tiles on demand, entirely on `localhost`.
* **Semantic Embeddings:** RemoteCLIP (ViT-B/32 backbone), 3-channel RGB, 224×224px input — **frozen as the production retrieval model** for the prototype. TerraMind was evaluated during research for its multi-band input capability but is explicitly outside the MVP implementation, documented as a research finding rather than a live alternative, so there is no ambiguity about which model the demo actually runs.
* **Vector Index:** **FAISS (HNSW)** — supports incremental addition without full index rebuilds. Retained deliberately after review: a filter-native store (sqlite-vec / LanceDB) was evaluated and **rejected** as an unnecessary replacement, because the documented FAISS weakness does not occur in this system's operating range (see "Metadata-filtered retrieval" below for the evidence and the in-place fix).
* **Catalog & Spatial Store:** SQLite, with the built-in **R-Tree module** for spatial bounding-box indexing — footprint-overlap lookups don't degrade to a full table scan as the catalog grows. Also supplies the metadata pre-filter that drives filtered vector search.
  **Concurrency configuration is mandatory, not tuning.** The FastAPI layer reads the catalog while a background worker writes change-detection results — SQLite's default rollback-journal mode makes readers and writers block each other and will throw `SQLITE_BUSY` under exactly this topology. Required settings on **every** connection: `PRAGMA journal_mode=WAL` (readers no longer block the writer, and vice versa), `PRAGMA synchronous=NORMAL`, `PRAGMA busy_timeout=5000`, `PRAGMA foreign_keys=ON`. WAL still allows only **one writer at a time**; a second concurrent writer gets `SQLITE_BUSY` immediately, which is why ingestion is serialized through a single worker.
  **`busy_timeout` alone does not cover every case:** a deferred transaction that reads and then upgrades to a write can take an immediate `SQLITE_BUSY` the timeout does not apply to — use **`BEGIN IMMEDIATE`** for any read-then-write transaction so the write lock is taken up front.
  **Connection-per-process, always opened after `fork()`.** SQLite is not fork-safe; a connection inherited across a `fork()` boundary can corrupt the database *without visible symptoms at the time*. Each worker process opens its own connection after being forked — never open in the parent and inherit. Also note Python's `sqlite3` opens transactions implicitly before DML and leaves autocommit off, so commit explicitly rather than relying on default isolation behaviour.
  **Constraint this places on deployment:** WAL does not work over network filesystems and requires all processes on the same host. The catalog must live on local disk — never on a network share.
* **Compute assumption (stated explicitly — it is a reported deliverable under §2.3):** RemoteCLIP inference is the dominant per-ingest cost, and embedding every crop is done **once** at ingestion, never at query time. Target hardware is a single workstation/laptop with a **CUDA-capable GPU**; RemoteCLIP runs on GPU where available and falls back to CPU otherwise. The fallback is functional but materially slower, and on CPU-only hardware embedding throughput — not change detection — becomes the ingestion bottleneck. Record the device actually used, and embedding throughput (crops/second), in the evaluation manifest rather than leaving hardware implicit. Retrieval is unaffected by this: query time is an index lookup with a single text/image embedding, which is fast on CPU.
* **Task Queue:** in-process Python `multiprocessing.Queue` — serializes ingestion so multiple dropped files don't get processed simultaneously and risk out-of-memory crashes. Deliberately not Celery/Redis: that's a production multi-user pattern requiring a second service to install and keep alive, unnecessary complexity for a single-analyst offline desktop tool.
  **Job state machine with crash recovery (required — a desktop app gets closed mid-job).** Every ingestion and change-detection job carries an explicit status column: **`queued → processing → completed`** (plus `failed`). Three rules make this safe:
  1. **Idempotency** — a unique constraint on the job's identity (for a change-detection job, the ordered pair `(scene_a_id, scene_b_id)`) with `INSERT ... ON CONFLICT DO NOTHING`, so the same pair can never be enqueued or processed twice.
  2. **Staging + atomic rename** — outputs (COGs, change rasters) are written to a staging path and moved into place with `os.rename` (atomic within a filesystem) only after the job succeeds. A crash therefore never leaves a half-written file that looks valid to the next run.
  3. **Startup reconciliation sweep** — on launch, any job still marked `processing` (i.e. it was interrupted) is reset to `queued`, and staged files with no committed catalog row are deleted. This directly resolves the crash-after-COG-before-catalog-entry case, which would otherwise leave orphaned files accumulating silently.
  The DB write that marks a job `completed` and the catalog row it produces go in the **same transaction**, so the two can never disagree.

## Six Required Capabilities → Where Each Is Implemented
1. **Semantic + image-to-image retrieval** — embedding pipeline + FAISS index + metadata-filtered search (AOI / date-range / sensor, per §2.2.1)
2. **Multi-temporal change analysis** — change-detection pipeline (below)
3. **False-alarm suppression / quality handling** — masking + normalization + confidence scoring (below)
4. **Discovery and clustering** — two mechanisms: per-query nearest-neighbour lookup ("Find Similar Sites"), plus periodic archive-wide clustering so groupings exist before anyone queries (§2.2.4 asks for *grouping across a wider area*, not only per-query neighbours)
5. **Analyst workflow and provenance** — review queue, confirm/reject, audit log (below)
6. **Scale, incremental ingestion, sovereignty** — incremental vector insertion (no full index rebuild), all processing local, COG ingestion for standard format support

---

## Ingestion Workflow 1 — Importing Dataset A (First Scene for an AOI)

1. **Universal Loader** — accepts `.SAFE`/JP2, GeoTIFF, or COG. Extracts bands.
2. **Capability Detection** — explicit audit stage immediately after loading: inspects band availability, QA/SCL band presence, native CRS, native resolution, before deciding the downstream processing path. Makes "works across sensors" a concrete, auditable decision point rather than implicit branching buried in code.
3. **Native-Resolution Quality Masking** — masks computed *before* reprojection or resampling, since interpolating a categorical mask (cloud/not-cloud) produces meaningless blended values at boundaries.
   - Sentinel-2 / Landsat: read SCL / QA_PIXEL band; Cloud, Cloud Shadow, Snow classes → 0. **SCL is the sole primary mask for L2A.** s2cloudless was previously carried as a supplement and is now demoted off the primary path: it is trained on **L1C** and documented not to perform well on L2A, which is what this pipeline ingests. Retained only as an optional secondary check if L1C input is ever used. This removes a CC-BY-SA-4.0 share-alike dependency from the critical path and one component from the MVP.
   - Bhuvan (4-band, no QA band): compute NDVI, NDSI (SWIR-based, degrade to NIR-based if unavailable), and Whiteness (mean R/G/B). A conservative spectral heuristic estimates potential cloud-like pixels: flagged if Whiteness > 0.25 AND NDVI < 0.2 AND NDSI < 0.2. **Framing matters:** this is an estimate, not reliable cloud detection — bright concrete, salt flats, and sand can trigger the same signature. It contributes to confidence weighting, never treated as a hard, trustworthy discard the way the pre-validated SCL/QA_PIXEL band is.
4. **CRS Reprojection** — happens after masks are generated; nearest-neighbor resampling for the categorical mask layer, bilinear/cubic for continuous band values.
5. **COG Conversion** — normalize to Cloud-Optimized GeoTIFF internally (required by the problem statement and by titiler). Record a checksum of the raw source file in the catalog before any deletion — deletion of the raw source is an explicit, logged, analyst-visible step, never a silent background action, given the submission's provenance/reproducibility requirement.
6. **RGB Composite Construction** — assembled from Red/Green/Blue bands (B04/B03/B02 for Sentinel-2) via GDAL. Never dependent on a sensor-provided convenience file (e.g. Sentinel-2's TCI) — one consistent code path across sensors.
7. **Three independent tiling concepts — deliberately not unified, stated explicitly to prevent confusion:**

   | Concept | Size | Purpose | Lifecycle |
   |---|---|---|---|
   | **Embedding crops** | Fixed 224×224px, resized | Input to RemoteCLIP | Generated once at ingestion, one per crop, feeds FAISS |
   | **Processing windows** | Variable, scene-level strips with 16–32px halo | Change-detection pipeline (Phases 1–5) | Generated per change-detection job, discarded after use except for the halo-trimmed output |
   | **Display tiles** | XYZ pyramid, size set by zoom level | Rendering in MapLibre GL | Generated on demand by `titiler`, never stored |

   These are not different names for the same thing — they solve different problems and have no reason to share a size. RemoteCLIP's 224×224 requirement is a fixed model input constraint (§ embedding pipeline). The halo/overlap concern below exists because change-detection connected-component analysis can truncate a real feature at a tile seam and lose it to the minimum-blob-size filter — embedding crops don't have this failure mode (a feature split across two search-index crops weakens its signal in each rather than being lost outright), so the halo fix applies to processing windows only, not to embedding crops.

   **Processing-window overlap detail:** each window is read with a surrounding halo of 16–32px (160–320m) that is discarded on write, so a feature straddling a seam is never truncated below the minimum-blob-size threshold. Halo width must exceed (morphological structuring-element radius + minimum-blob radius). Polygons touching a shared window edge are merged by **best-match** (highest shared boundary length), never naive transitive merging. The failure mode being avoided: naive transitive merging chains every touching polygon together, so a sequence of adjacent boundary contacts can cascade into one runaway segment spanning a large part of the scene. Best-match merging bounds this by letting each boundary segment join only its single highest-contact neighbour. **Preferred implementation:** `rasterio` overlapping strip windows (full width, banded height, with halo) aligned to the file's internal block structure, so seams occur in one dimension only. Two practical notes: a windowed read of a *non-internally-tiled* GeoTIFF still loads the whole image, so re-tile on COG conversion (`tiled=True, blockxsize=512`); pin a rasterio version clear of the windowed-read memory leak reported in 1.4.2.
8. **Embedding Pipeline** — RGB **embedding crops** (224×224, per the table above — not the processing windows from step 7) → RemoteCLIP → vector embeddings, added incrementally to the FAISS HNSW index. Powers both text-to-image semantic search and image-to-image similarity ("find similar").
   **NaN handling on this path is mandatory, and is a silent-failure risk rather than a crash risk.** The Cleaned Baseline from Phase 1 carries NaN over masked pixels, and unlike ECC/PCA/K-Means — which raise loudly — **a ViT forward pass propagates NaN without error**. The result is a NaN embedding vector inserted into the FAISS HNSW graph, where distance computations against it return NaN, corrupting graph traversal and returning garbage rankings with nothing raised anywhere in the system. A single cloud-edge crop can quietly poison the search index. Two required guards, applied before any crop reaches the model:
   1. **Minimum valid-pixel gate** — a crop below a minimum valid-pixel fraction is **skipped entirely and never embedded**. An unusable crop must be *absent* from the index, not represented by a fabricated vector.
   2. **Fill before forward pass** — for crops that pass the gate, replace residual NaN with a neutral value before inference, so no NaN can enter the model under any circumstance.
   Additionally, assert every produced vector is finite before insertion into FAISS — a cheap last line of defence against this failure class.
9. **Catalog Write** — SQLite record per tile: tile_id, scene_id, sensor, acquisition_date, geo-bounds, **MGRS grid reference** (10-digit, via the `mgrs` Python package — MIT licence, wraps NGA's GeoTrans; one call per coordinate), COG path, cloud_pct, and `faiss_id` (the vector's position in the FAISS index — the vector itself lives only in FAISS, never duplicated into SQLite). Spatial bounds indexed via the R-Tree module. This table is what produces the ID allowlist for filtered search.
**Pipeline Visibility (UI feature).** The right-side "Workspace" panel shows a live vertical step list of the ingestion pipeline as it runs: Capability Detection → Masking → Reprojection → COG Conversion → Tiling → Embedding → Indexing, each with an icon state (pending / in-progress / done), live counts where applicable (e.g. "1,204 / 3,000 tiles embedded"), and the current stage highlighted. When idle, the panel shows "No active pipeline" rather than blank space. This exists so the analyst sees real progress rather than a spinner, and so a judge watching the demo sees the pipeline architecture in action rather than a loading bar. Not a separate subsystem — it reads job status from the existing job-state-machine table in SQLite.

10. **Pipeline Visibility Panel** — the right sidebar ("Workspace") shows live pipeline progress during ingestion: which stage is currently running (capability detection → masking → reprojection → COG → tiling → embedding), with per-stage status (pending / in-progress / complete), live tile counts where applicable (e.g. "1,204 / 3,000 tiles embedded"), and elapsed time. This is not a loading spinner — the analyst sees which processing path was chosen (SCL masking vs Bhuvan heuristic, which bands detected, which CRS) and can watch the pipeline advance stage by stage. On completion, the panel shows a summary of what was ingested. Serves two purposes: (a) the analyst knows the system is working, not frozen, and (b) during a live demo, judges can see the architecture's stages actually executing rather than trusting a diagram.
11. **Asynchronous Availability** — the scene is viewable on the map (via titiler + MapLibre) as soon as its COG and tiles exist, without waiting for embedding or change-detection background processing to finish. Only search and change-candidate results lag behind until their respective background jobs complete — the analyst is never staring at a blank window during indexing.

## Ingestion Workflow 2 — Importing Dataset B (Subsequent Overlapping Scene)

Steps 1–9 above run identically and independently for Dataset B. At catalog-write time, the system checks for spatial footprint overlap with existing entries via the R-Tree index.

**Same-sensor constraint on pairing (hard rule).** Change-detection pairing is restricted to **scenes from the same sensor**. Footprint overlap alone is not sufficient grounds to difference two scenes. Sentinel-2, Landsat C2 and LISS-III differ in spectral response function and view geometry; NASA's Harmonized Landsat Sentinel-2 (HLS) product exists precisely because making those sensors interchangeable requires BRDF normalisation (c-factor, Roy et al. 2016) and spectral bandpass adjustment — **neither of which this pipeline performs**. Differencing across sensors without that harmonisation presents sensor difference *as change*, producing exactly the confident false positives the entire Phase 1–3 suppression stack exists to prevent. Cross-sensor data remains fully available for ingestion, retrieval, browsing and display — **only differencing is restricted.** Recorded as a stated limitation, not a capability claim. (Ingesting the HLS product directly would lift this constraint; out of MVP scope.)

**Comparison-pair selection (automatic default, no analyst decision required):** if Dataset B overlaps an existing scene **from the same sensor**, it is paired against the **nearest chronological prior observation** covering the same footprint — not against every historical overlapping scene, which would grow combinatorially as the catalog fills (100 indexed scenes for one AOI would mean up to 4,950 possible pairs if done naively; this design deliberately avoids that). **Optionally, if a same-season prior-year observation also exists** (reusing the same "same calendar month, prior years" lookup used for persistence filtering in Phase 5 below), a second comparison against that seasonal reference runs alongside the chronological one, catching cases the chronological-only comparison might miss due to phenology mismatch. This works immediately for a newly indexed AOI (only the chronological comparison needs a second scene to exist at all) and strengthens automatically as more history accumulates — an analyst-selectable manual override profile is a reasonable future addition, but not required for the system to function correctly from the first import onward.

---

## Change Detection Pipeline (False-Alarm Suppression Engine)

Runs as a background job triggered at ingestion time (Workflow 2 above), not at query time — see "Query-Time Performance" below.

### Phase 1 — Quality Masking & Trust Scoring
- **AOI Cloud Trust** — cloud cover computed strictly within the analyst's AOI, not the whole scene. Rather than a hard reject-and-discard, assign a graded `CloudTrust = 1 − cloud_fraction` (e.g. 40% cloud → 0.60 trust) that scales downstream confidence. A hard reject risks leaving an analyst with nothing at all for a persistently cloudy AOI across every available date; graded trust is consistent with how every other quality signal in this pipeline works — nothing is binary-discarded, everything degrades confidence proportionally.
- **Mask Application** — invalid pixels become NaN, establishing the "Cleaned Baseline." This cleaned version feeds both the embedding pipeline and change detection, not change detection alone.
- **NaN-handling contract (mandatory — every numerical stage downstream rejects NaN).** Masking produces NaN, and none of the libraries in this pipeline tolerate it. `cv2.findTransformECC` computes `rho = correlation/(imgNorm*tmpNorm)` and raises `CV_Error(Error::StsNoConv, "NaN encountered.")` on a single NaN pixel; `sklearn.decomposition.PCA` and `sklearn.cluster.KMeans` both call `_assert_all_finite` and raise `ValueError: Input contains NaN...`. NumPy masked arrays are *not* understood by scikit-learn either. The required pattern, applied at every numerical stage:
  1. Carry the boolean **validity mask** alongside the raster as a first-class array — never rely on NaN alone to signal invalidity.
  2. **Extract valid pixels** into a flat 2-D array (`X[valid]`) before PCA/K-Means.
  3. Run the algorithm on that flat array only.
  4. **Scatter results back** into a full-size raster initialised to an explicit nodata/unlabelled value.
  **Do not impute NaN → 0 in a difference image.** Zero is a *meaningful* value there (no change), so imputing invalid pixels to zero silently relabels unobserved ground as verified no-change — a correctness failure, not a cosmetic one.
  For ECC specifically, masking is **not sufficient on its own**: replace NaN with a fill value (0 or local mean) *and* pass the validity mask as `inputMask`, since NaN left in the buffer propagates through the projection arithmetic regardless of the mask. Pass `gaussFiltSize` explicitly (`None, 5`) — several OpenCV builds reject a `None` mask positionally.

### Phase 2 — Geometric Alignment
**Ordering is load-bearing in this phase: the Grid Check runs FIRST.** A logical AND of two masks is only meaningful once both rasters are on an identical pixel grid — ANDing masks that are not yet pixel-aligned combines values from different ground positions and produces a mask that is silently wrong rather than erroneous. An earlier version of this document listed the mutual mask before the grid check; that order is incorrect and must not be reintroduced.

- **1. Grid Check (hard gate — nothing downstream in this phase runs until all four match)** — verify **CRS, pixel origin, affine transform, and resolution** are all identical between Image A and Image B. Matching CRS alone is *not* sufficient: two scenes in the same UTM zone can have pixel-grid origins offset by a non-integer number of pixels, and differencing them then compares partially-overlapping ground area, injecting spurious edge change along every high-contrast boundary. Note also that GDAL's `-tap` aligns an output extent to the resolution grid but does **not** guarantee two scenes share a grid — so verify explicitly by comparing `src.crs`, `src.transform`, and `src.shape` for equality, and warp the moving image onto the reference grid where they differ. Sub-pixel residual *after* grid alignment is what ECC removes; grid misalignment must be fixed by resampling first and cannot be fixed by co-registration.
  **Related ingestion rule:** adjacent Sentinel-2 tiles can arrive in *different UTM zones*, so an AOI straddling a zone boundary would otherwise never stack. Pick one target CRS per AOI (the AOI centroid's UTM zone) at ingest and pin every acquisition for that AOI to it.
  **On failure:** the moving image is warped onto the reference grid and the check is re-run. If it still fails (e.g. the pair has insufficient true geographic overlap after warping), the pair is rejected with an explicit `grid_mismatch` status rather than proceeding — never differenced on a best-effort basis.
- **2. Mutual Validity Mask** — logical AND of Mask A and Mask B, computed **only after the Grid Check above has passed**. Standard alignment algorithms fail on NaN input, so this is required before alignment, not optional.
- **3. Terrain-Risk Masking (DEM-based)** — slope computed from CartoDEM (hosted on Bhuvan/NRSC, 30m posting — no new data source to justify). Steep slopes receive reduced confidence, since viewing-angle differences cause terrain-induced illumination/shadow shifts that look like change but aren't. **Functional form, specified rather than left as design intent:**

  ```
  TerrainFlatness = cos(slope_angle),  floored at cos(60°) ≈ 0.50
  ```

  Cosine is the principled choice rather than an arbitrary decay curve: both view-angle foreshortening and the illumination geometry driving apparent change on slopes are cosine relationships in the incidence angle, so the confidence multiplier follows the same geometry as the error it discounts. Flat ground → 1.0; 30° → ≈0.87; 45° → ≈0.71. At and beyond 60° the term **floors** at ≈0.50 rather than decaying toward zero, because very steep terrain should heavily discount a detection, not silently annihilate it — such candidates are additionally flagged for manual review instead of being scored out of existence. Floor angle, and whether to floor at all, are calibration parameters. **Fallback:** if CartoDEM coverage is unavailable for a given AOI, the terrain-flatness confidence term defaults to neutral rather than failing the pipeline.
- **4. Masked Co-Registration** — primary: `cv2.findTransformECC()`, computed over the mutual validity mask with NaN pre-filled (see the NaN contract in Phase 1). **Both failure modes throw rather than returning a status flag, so the call must be wrapped in try/except:** `cv2.error` "NaN encountered." (StsNoConv) and `cv2.error (-7)` "The algorithm stopped before its convergence" on non-convergence. Either exception escalates to the ORB fallback rather than crashing the job.
  **`rho` gate — the silent failure path, and the one that matters most here.** Exceptions are the *loud* failure modes. ECC can also **converge to a bad local optimum and return a transform with no exception raised at all** — most commonly over low-texture regions (open water, bare fields, uniform desert) where the correlation surface is nearly flat, and on displacements larger than the algorithm's basin of convergence. A silently bad transform injects false change across the entire scene pair, and nothing downstream would attribute the result to alignment. Therefore: **read the correlation coefficient `rho` that `findTransformECC` returns and threshold it** (starting value ~0.8, a calibration parameter). Three outcomes:
  - `rho` above threshold → accept the transform, proceed.
  - `rho` below threshold → **treat exactly as a failure**, escalate to the ORB+RANSAC fallback. Do not accept a low-`rho` transform merely because no exception was thrown.
  - ORB fallback also below its Inlier-Ratio threshold → reject the pair with an explicit `alignment_failed` status and surface it for manual review, rather than differencing on an alignment nothing trusts.
  This gate is independent of, and runs before, the RMSE > 0.5px check and the block-level CGRA scoring — those characterise *how good* an accepted alignment is; the `rho` gate decides whether it should be accepted at all. For large initial displacements, use a Gaussian-pyramid coarse-to-fine initialisation rather than expecting single-scale ECC to converge.
  **Evidenced alternative primitive (optional, adopted on measurement only).** FFT **phase correlation** per block — rather than intensity-gradient optimisation — is reported in the co-registration literature (AROSICS; Storey/Yan on Landsat-8 / Sentinel-2A sub-pixel registration) as more robust than ECC to land-cover change and cloud, and it produces a natural per-block confidence measure (correlation-peak sharpness ratio plus forward-backward consistency) that feeds the block-level scoring in step 5 directly rather than being bolted on afterwards. Displacements from high-confidence blocks fit a dense warp (thin-plate spline / RBF); low-confidence blocks become the uncertain-zone mask. This is a **primitive substitution, not a redesign** — every confidence rule above (graded trust, no binary rejects, block-level rather than scene-level scoring, uncertain zones raising the evidence bar) applies unchanged. **Not adopted as mandatory:** the ECC + ORB/RANSAC path is technically sound, fully failure-handled, and already specified to implementation depth. Phase correlation is recorded as a justified upgrade to evaluate if time allows, and should be adopted on **measured registration accuracy against a labelled pair set** — never on the assumption that a different primitive is inherently better. Use `MOTION_EUCLIDEAN` (or `MOTION_TRANSLATION`) rather than `MOTION_HOMOGRAPHY` — since scenes are already on a common grid, residual is sub-pixel to a few pixels, and homography's extra parameters diverge more readily. Fallback if ECC fails or RMSE > 0.5px: ORB feature detection + RANSAC, graded by Inlier Ratio (matches surviving RANSAC ÷ total ORB matches). DEM terrain masking remains necessary regardless — feature alignment alone doesn't solve terrain-induced mismatch in mountainous areas.
- **5. Block-Level Alignment Confidence (CGRA)** — rather than one whole-image alignment score, score confidence per small overlapping block (match sharpness, forward-backward consistency, visible detail). Low-confidence blocks are marked an "uncertain zone" and cannot trigger a change alert alone — detections inside require stronger evidence. Rendered via the same green/yellow/red reliability-map visualization used elsewhere in the confidence system.

### Phase 3 — Radiometric Normalization
- **Temporal Median Compositing (optional)** — where multiple same-location dates are indexed, per-pixel median (NumPy/Xarray) across them suppresses residual shadow/cloud noise the mask alone might miss. Local processing only — never routed through a cloud API (e.g. Google Earth Engine), which would break the offline constraint.
- **TASC (Topographic/Sun-Angle Correction)** — using CartoDEM and Image B's solar azimuth/elevation metadata (included in product metadata, no external dependency), compute local solar incidence angle per pixel and apply a Minnaert correction to balance shaded vs. brightly lit terrain. Spatially-varying, not a single global adjustment.
- **PIF (Pseudo-Invariant Features)** — anchor pixels selected on a **joint criterion**, not variance alone: (1) low temporal variance across catalog history, and (2) high local block-level registration confidence (from CGRA above). The second condition matters because a site under slow, ongoing change (e.g. a construction site mid-build) can appear artificially stable across a limited observation window on variance alone, while genuinely evolving texture also tends to produce weaker feature-matching confidence — combining both filters out more false-stable anchors than either alone. Selection is statistical, not a hardcoded material list — concrete runways and deep water are common examples, not the criterion itself, since austere terrain (high-altitude, dense jungle canopy) may have neither. **Fallback order:** (1) apply the joint criterion; (2) if too few anchors found, skip PIF for this pair and rely on TASC alone; (3) record which tier ran — a detection processed without PIF carries a lower confidence contribution from radiometric normalization than one with solid anchors. Compute mean brightness of selected PIFs in both images; fit a linear regression multiplier; apply to Image B to match Image A's radiometric baseline.

### Phase 4 — Change Detection Engine

**Which bands are differenced (previously ambiguous — now specified).** Do **not** stack all 13 Sentinel-2 bands: adjacent bands are highly correlated, and hyperspectral change-detection research (Remote Sens. 2017, 9, 1008) shows this redundancy "inevitably lead[s] to information redundancy, thus reducing the sensitivity and accuracy of the CD process." The 60m atmospheric bands (B01/B09/B10) add noise and nothing else. Use:
- **Change magnitude:** the four 10m bands (B02/B03/B04/B08), optionally spectral-PCA'd to 1–2 components first to decorrelate and denoise.
- **Primary PCAKM input:** a **single** difference image (NIR difference is the strong default — maximum vegetation/built-up contrast). This matches how the method actually works: Celik's original PCAKM (IEEE GRSL 6(4):772–776, 2009) partitions *one* difference image into h×h non-overlapping blocks, extracts eigenvectors from that block set, projects each pixel into eigenvector space, and runs k-means (k=2) on those feature vectors. Its "PCA" is a **local-neighbourhood feature extractor on one difference image**, not a spectral decomposition of a multi-band stack — a distinction easy to get wrong in implementation.
- **Direction channels:** signed ΔNDVI / ΔNDBI / ΔNDWI (see direction classification below) — used for labelling and false-alarm filtering, not as magnitude inputs.

**Steps:**
- **Subtraction** — ΔI = Image B − Image A on the chosen difference channel(s), aligned and normalized. **Gated by the Phase 2 Grid Check** — subtraction must not run unless CRS, origin, transform and resolution are verified identical.
- **Block-PCA** — per Celik, on the difference image, run on the **extracted valid-pixel vector only** (per the NaN contract in Phase 1), never on the raster with NaN in place. Retained component count set empirically during OSCD validation, never stated as a hardcoded a-priori percentage.
- **K-Means Clustering** — K=2, on the same extracted valid-pixel vector; labels are then scattered back into a full-size raster initialised to an explicit unlabelled value.
  **Cluster-label resolution rule (mandatory — the cluster IDs are arbitrary).** K-means returns two unlabelled clusters; *which one means "changed" is not fixed and will silently invert between scenes* if read from the cluster index. Resolve it explicitly every run: **the change cluster is the one with the larger centroid magnitude** (greater L2 norm in feature space), since difference-image values where change occurred are higher than where nothing changed. In practice this is also usually the smaller cluster by pixel count, which serves as a sanity check — but centroid magnitude is the rule, not pixel count. Never assume cluster index 1 is change.

**Known characteristic to design around:** PCAKM is documented as noise-sensitive and false-alarm-prone (SAR studies report false-alarm rates up to ~39%; the Sensors 2022 comparative study built 22 modified variants specifically to reduce this). This is precisely why the false-alarm suppression stack in Phases 1–3 and the direction cross-checks below are load-bearing, not optional polish.

### Phase 4b — Change Direction Classification (appearance / disappearance / expansion / contraction)

**Why this phase exists:** §2.2.2 explicitly requires identifying "appearance, disappearance, expansion or contraction of features." K-means clusters the *magnitude* of difference features and therefore **discards sign entirely** — a binary mask can never answer "did this appear or vanish." Direction must be reconstructed from an auxiliary signed quantity. This was a genuine coverage gap in earlier versions of this architecture.

**Mechanism (training-free, three signals combined):**
**Terminology — two distinct connected-component sets exist in this pipeline and must not be conflated:**

| Name | Derived from | Used for | Where |
|---|---|---|---|
| **Classification Components** | Each date **independently**, via index-threshold pseudo-classification | Direction classification (appearance / disappearance / expansion / contraction) | Phase 4b (here) |
| **Change Blobs** | The **binary change mask** from Phase 4 | Morphological cleanup, shape classification, confidence scoring | Phase 5 |

**Execution order (fixed):**

```
Classification Components → Direction Classification
        → Binary Change Mask → Opening → Closing → Confidence Scoring
```

**Morphological cleanup operates on Change Blobs only and never modifies the Classification Components.** Direction classification completes before any morphology runs, so opening and closing cannot alter the per-date objects that appearance/disappearance/expansion/contraction are derived from.

1. **Classification Component matching** — label connected components in each date independently using index-threshold pseudo-classification. **Named calibration parameters (an earlier draft used a single undefined identifier `threshold` for two different comparisons — not implementable):**

   | Class | Rule | Parameters |
   |---|---|---|
   | Built-up | `NDBI > NDBI_BUILT_MIN` AND `NDVI < NDVI_BUILT_MAX` | `NDBI_BUILT_MIN` (start 0.0), `NDVI_BUILT_MAX` (start 0.2) |
   | Water | `MNDWI > MNDWI_WATER_MIN` | `MNDWI_WATER_MIN` (start 0.0) |
   | Vegetation | `NDVI > NDVI_VEG_MIN` | `NDVI_VEG_MIN` (start 0.3) |

   All five are calibration parameters, not constants — starting values are domain-standard defaults only. **Index thresholds are strongly scene-dependent**, so prefer **per-scene Otsu thresholding** over these fixed values wherever the histogram supports it, falling back to the defaults when it does not.

   **Band definitions and the 10m/20m resampling rule (previously unspecified — a real implementation trap):** NDVI = (B08−B04)/(B08+B04), both native 10m. NDBI = (B11−B08)/(B11+B08) and MNDWI = (B03−B11)/(B03+B11) — **B11 (SWIR) is 20m while B08/B03 are 10m.** Required handling: **resample B11 up to the 10m analysis grid using bilinear interpolation** (never nearest-neighbour — a nearest-neighbour 20m block edge can shift by a whole 20m cell between two acquisitions, manufacturing false NDBI change along every edge), onto the **identical shared reference grid for both dates** (per the Phase 2 Grid Check). Upsampling adds no real detail and the effective resolution of any B11-derived index remains 20m — state this rather than implying 10m precision for NDBI/MNDWI. Use **MNDWI (Xu, B03/B11) rather than McFeeters NDWI (B03/B08)** for water extent: SWIR suppresses the built-up signal that the NIR-based form misclassifies as water.

   **Known limitation to carry into the output, not suppress:** NDBI confuses **bare soil with built-up** — a well-documented failure, and directly relevant to agricultural-to-urban change in India, where freshly-ploughed or fallow fields will fire as "new built-up." Flag NDBI-driven built-up detections in agricultural contexts as **soil-ambiguous** rather than reporting them as confirmed construction.

   Then match components between dates by spatial overlap at **IoU ≥ 0.5** (the standard positive-match criterion across the detection/segmentation and change-detection literature), with **centroid distance** as tie-breaker for small objects where IoU is unstable. Then:
   - present in B only → **appearance**
   - present in A only → **disappearance**
   - matched, area grew/shrank → **expansion / contraction**, requiring the area delta to exceed co-registration error (1–2px = 10–20m) plus a relative tolerance (~±20%), so alignment jitter never reads as growth.
2. **Signed spectral-index deltas** — mean ΔNDVI/ΔNDBI/ΔNDWI *inside each blob* name the change: ΔNDBI>0 & ΔNDVI<0 → vegetation→built-up (construction); ΔNDVI>0 → regrowth/greening; ΔNDWI>0 → new water/flooding. This is the most robust and most explainable single signal, and it needs no training.
3. **Change Vector Analysis (CVA) cross-check** — 2-band CVA (NIR vs Red) yields per-pixel change magnitude *and* direction angle; direction-dominated CVA (DCVA, Chen et al., Int. J. Applied Earth Obs. Geoinf.) determines change type from the vector angle first, then applies a per-type magnitude threshold, and has been demonstrated on Sentinel-2A for "from-to" change extraction. Used here as a confidence cross-check, not as the primary classifier.

**Required caveat:** CVA carries a known ambiguity — different real-world changes can share the same vector angle (Carvalho et al., Remote Sens. 2011, 3(11):2473). Direction labels therefore always carry a confidence score and are never presented as certain.

### Phase 5 — Cleanup, Seasonal Filtering, and Scoring
- **Morphological Cleanup (operates on Change Blobs only — never on Classification Components) — two distinct compound operations, in this order.** An earlier version of this document said "erosion then dilation" while claiming both noise removal *and* hole filling; that is wrong and would silently deliver only half the stated effect. Erosion-then-dilation is an **opening** (removes small bright objects, approximately preserves the size of large ones); dilation-then-erosion is a **closing** (fills small holes and gaps). Achieving both requires **four operations, not two**:
  1. **Opening** (erode → dilate) — removes 1–2px salt-and-pepper noise.
  2. **Closing** (dilate → erode) — fills small internal gaps (e.g. a hole inside a detected building's roof).
  Structuring element: **3×3 square, or a disk of radius 1**, at 10m resolution. Larger elements erase genuine small changes. Side effects to respect, both real: **opening can sever thin linear features**, and **closing can merge nearby distinct Change Blobs** — the latter distorts blob counts and areas used for shape classification and confidence scoring in this phase. It cannot affect direction classification, which ran in Phase 4b on Classification Components and is complete before any morphology executes.
  **Ordering constraint with linear-feature classification (later in this phase):** because opening severs thin features, the linear-feature shape-descriptor path must run on the **pre-opening Change Blobs**, or on a separate branch that skips the opening entirely. Running shape descriptors on the post-opening mask would systematically destroy the very features that classification is trying to identify. Prefer **connected-component area filtering (minimum mapping unit)** over aggressive morphology for the linear branch: at 10m, an MMU on the order of ~10 pixels (≈0.1 ha) is the conventional noise filter. Note the genuine detectability limit this exposes — a single new building footprint may be only 1–4 pixels and will be filtered out. That is a resolution limit, not a bug.
  Kernel size and MMU are calibration parameters (see Calibration & Validation Strategy).
- **Persistence Filter** — runs on **every** surviving Change Blob whose mean NDVI drop exceeds **`NDVI_DROP_SIG`** (a named calibration parameter; starting value 0.15, set during OSCD validation), not gated by a geometry pre-filter (an earlier version only checked "organic-shaped" polygons, which would let a real clearance event with straight field boundaries skip verification — fixed). For each candidate: query the catalog for the same AOI, same calendar month, across whatever historical observations are available (not a fixed 1–2 year window — a newer AOI with less history still gets the best check possible, strengthening automatically as more years accumulate). Cyclical drop → discard as Seasonal. Step-change breaking historical persistence → retain as Clearance. Geometry can be recorded as supporting evidence, never as the gate deciding whether the persistence check runs.
- **Confidence Scoring** — computed per detected candidate polygon, **after** morphological cleanup, so isolated noise pixels can't dominate a candidate's score:

```
Confidence = 0.35 × NormRMSE + 0.35 × NormClusterDist + 0.15 × TerrainFlatness + 0.15 × ValidCoverage
```
  All four terms explicitly normalized to [0,1] before combining — RMSE and cluster distance are not naturally bounded the way coverage is, so each needs its own normalization step, not just the label "normalized" on one term. **`NormRMSE`:** a saturating map, `1 − min(RMSE / RMSE_max, 1)`. **`NormClusterDist`:** the candidate's **per-scene percentile rank within that scene's changed cluster** — this is part of the specification, not an implementation detail. Cluster distance is unbounded and scene-dependent, so any fixed normalisation would make confidence scores non-comparable between scenes, silently breaking review-queue ranking (candidates from different scenes sit in one queue). Percentile rank is bounded to [0,1] by construction and comparable across scenes. Weights are illustrative starting values, stated in any submission as "initial weights calibrated during validation against OSCD," never as derived constants.
- **Mutual Valid Coverage Hard Floor** — if a candidate polygon's mutual valid coverage falls below a calibrated minimum (illustrative starting point: 30%, to be set during OSCD validation), it is dropped with an explicit **"Insufficient Evidence"** flag rather than being scored and potentially passed through on a thin data sliver. This is a hard gate, separate from and in addition to the weighted ValidCoverage term above — a detection shouldn't be allowed to reach a moderate confidence score purely because the other three terms happened to look good on the small fraction of the tile that was actually valid.
- **Earliest-Observation Estimate** — walk backward through catalog timestamps, **skipping any candidate date whose valid-pixel coverage falls below the minimum threshold** — a heavily cloud-masked earlier scene should never be reported as the "earliest observation" simply because it's the oldest date on file.
- **Spectral + Shape Classification** — NDVI/NDWI/NDBI classify surviving changes as vegetation clearance / water-extent change / construction. NDSI (graded, not just SCL's binary flag) flags snow; snow-covered regions default to unreliable/noise for land-surface queries.
  **Linear features (roads) — scope corrected, and this matters for what we claim.** Spectral indices give no shape information, so NDBI alone cannot separate a new road from a new building. Shape descriptors computed on each connected component do: **elongation** (MBR aspect ratio), **eccentricity** (principal-axis eigenvalue ratio), **compactness** `P²/(4πA)`, **solidity** (area ÷ convex-hull area), and **skeleton-length-to-area ratio** (high for thin linear features). Operational rule, requiring multiple descriptors in agreement since single descriptors disagree across definitions: elongation > 3 AND eccentricity > 0.9 AND solidity < 0.5 → linear. Calibrate exact cutoffs on our own scenes.
  **However — road development is NOT claimed as a verified capability at 10m, deliberately.** Ayala, Aranda & Galar (ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci., V-3-2021, 9–14) state plainly that "the feasibility to detect a road depends on its width, which can reach sub-pixel size in Sentinel-2 imagery," and reach usable road maps only by *super-resolving to 2.5m* — plain 10m U-Net gave IoU ≈37%, rising to ≈69% only after super-resolution. Google Research's 2024 Sentinel-2 building/road work reaches 50cm-equivalent masks only via a distilled teacher-student model trained on high-resolution labels — a heavy trained model we explicitly do not have, and whose reported figures come from supervised deep learning that our unsupervised classical pipeline will not match. **Therefore:** elongated change blobs are surfaced as low-confidence **"possible new linear feature (road, cleared corridor, pipeline)"** with shape descriptors and ~10–20m positional uncertainty shown. Confident linear calls are reserved for features ≥~30m wide (major highways/interchanges) or corroborated across multiple dates. Claiming road-network mapping at 10m without a trained super-resolution model is not defensible and will not survive a technical question.

**S2SR super-resolution:** display-only if used at all. Never feed super-resolved output into any step above — it hallucinates plausible detail rather than recovering real information, a liability for a defense evaluation. The same principle applies to any AI-based cloud/haze "removal" or inpainting: the system never attempts to reconstruct what's hidden under cloud, haze, or snow — it retrieves and compares against genuinely clearer observations from other dates instead, never a plausible-looking guess.

**SAR scope (explicit boundary):** Sentinel-1 ingestion (Lee Sigma speckle filtering, Local Incidence Angle layover/shadow masking via CartoDEM) is an isolated, deferred pipeline — not part of the default path, triggered only by explicit analyst override when optical fallbacks are exhausted.

---

## Discovery & Clustering (§2.2.4)

Two complementary mechanisms, because the requirement asks for *grouping across a wider area* — not only per-query neighbours:

1. **Per-query "Find Similar Sites"** — nearest-neighbour lookup against the embedding index from any retrieved or confirmed result.
2. **Archive-wide clustering, precomputed** — periodic unsupervised clustering over stored embeddings so groupings exist before anyone asks, exposed as a browsable facet. **Method: K-means with silhouette-selected *k*, on L2-normalized embeddings** (spherical k-means — valid on unit vectors, since cosine and Euclidean distance are monotonically related there).

   **Why K-means and not UMAP→HDBSCAN.** The BERTopic-style pipeline (L2-normalize → UMAP(cosine) → HDBSCAN) was evaluated and deferred for two reasons. First, **explainability**: a stochastic nonlinear projection feeding a density-based clusterer is difficult to defend when a reviewer asks *why* two sites were grouped — and explainable processing is a hard constraint on this system, not a preference. K-means centroids answer that question directly. Second, **dependency weight**: UMAP and HDBSCAN are two substantial additions for a capability K-means already delivers, and K-means is already in the stack (Phase 4). Determinism is a third benefit — the same archive produces the same groupings across runs, which matters for a reproducible evaluation report.

   *Known trade-off, stated honestly:* K-means requires choosing *k* and assigns every tile to some cluster, so it has no native "this site belongs to no group" outlier state. Silhouette selection mitigates the first; the second is accepted for MVP.

---

## Analyst Review Queue & Audit Trail

Satisfies the problem statement's analyst-workflow and provenance requirement (capability 5):
- **Candidate actions** — analysts confirm or reject any detected change candidate polygon in the UI.
- **SQLite persistence** — every decision stored locally with timestamp, local operator identifier, and optional notes.
- **Visual Evidence Card** — clicking any confirmed detection generates a single-page printable HTML summary: before/after thumbnail chips, MGRS coordinate, change type and direction, confidence with four-term decomposition, sensor metadata, acquisition dates, SHA-256 raster checksums (already in the catalog), and analyst sign-off with timestamp. One HTML template populated from existing provenance fields — not a separate generation pipeline.
- **Exportable history** — the complete decision history and audit trail exports locally for operational reporting. **Format: GeoJSON** (with explicit CRS declaration), one feature per change candidate, each carrying a provenance block: source scene IDs for both dates, acquisition timestamps, sensor, processing-stage record (which masking path, which PIF fallback tier, alignment method and RMSE, retained PCA components), confidence score with its four component terms, direction classification with its confidence, and the analyst decision with timestamp and operator identifier. This satisfies §2.2.5's requirement that "exported results must retain source-scene and processing provenance" concretely rather than by assertion.

**Visual Evidence Card export.** Beyond the raw GeoJSON, clicking any confirmed detection generates a single-page printable HTML summary: before/after thumbnail chips, MGRS coordinate, change type and direction, confidence score with four-term decomposition, sensor metadata, acquisition dates, SHA-256 raster checksums (already in the provenance manifest), and analyst sign-off. This is a **formatted view of data the system already stores**, not a new computation — one HTML template populated from existing provenance fields. Its value is operational: it produces a standardised intelligence summary in a format suitable for passing up a command chain, not a data dump that requires GIS software to read.

**Implementation roadmap for this section:**
- **Confirm/reject actions, SQLite persistence and GeoJSON export** — in MVP scope, built alongside the review queue.
- **Automated evaluation-report generation** (latency / storage / scene-count metrics) — **in MVP scope, and instrumented from the first commit**, per Evaluation Instrumentation below. Build time and query latency cannot be reconstructed retroactively, so the logging goes in as the pipeline is written, not as a reporting step bolted on at the end.
- **Feedback-driven reranking** (using confirm/reject history to improve future ranking) — **deferred beyond MVP.** The decision history is captured now so the capability can be added later without re-collecting data; the reranking logic itself is not built in this version.

---

## Query-Time Visual Attribution (GAE Heatmaps)

**Problem solved:** semantic search returns a tile covering ~5 km² — the analyst still has to find the relevant feature within that box manually.

**Mechanism:** Generic Attention-model Explainability (Chefer, Gur & Wolf, ICCV 2021 — "Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers"). For each top-K retrieval result, one backward pass through the RemoteCLIP ViT computes gradient-weighted attention rollout from the text query to the image patches, producing a per-patch relevance heatmap. Official implementation: `hila-chefer/Transformer-MM-Explainability`, MIT licence. Confirmed working with CLIP ViT-B/32.

**What it produces:** a 7×7 relevance grid over the 224×224 crop (ViT-B/32's patch size is 32×32 pixels). At 10m/px, each patch covers ~320m × 320m, so the heatmap localises the query-relevant region to **~0.1 km² per highlighted patch — roughly a 50× reduction** in the area the analyst needs to inspect, from ~5 km² to ~0.1 km².

**Precision claim, stated honestly:** this is patch-level localisation (~320m), not pixel-level. Do not overclaim as "sub-hectare" (a hectare is 100m × 100m; the patches are larger).

**When it runs:** at **query time only**, on the top-K returned results. One backward pass per result — heavier than a pure lookup but bounded by K, and runs on GPU. Does not affect ingestion throughput.

**UI rendering:** semi-transparent colour overlay ("attribution glow") on top of the retrieval tile in the results panel.

---

## Known Scale Limitation — Tile Size vs. Tactical Targets

A 224×224px **embedding crop** (per the tiling table in Ingestion Workflow 1) at Sentinel-2's 10m/px resolution covers roughly 2.24km × 2.24km of ground. If that tile contains a village, a river, a forest, and one small structure of actual interest, the embedding summarizes the tile by its dominant content — small features get diluted rather than surfaced.

**Multi-scale sub-tiling (32×32/64×64px crops upsampled to 224px) was evaluated as a mitigation and is explicitly cut from MVP scope** — upsampling that aggressively risks interpolation artifacts that degrade embedding quality for a Vision Transformer rather than improving small-object retrieval, since ViT patchifies its input and heavily-interpolated patches don't resemble what the model was trained on. This reverses an earlier "attempt if time allows" framing — worth stating plainly in the submission as a deliberate scope decision with a real technical reason, not a silently dropped feature.

**Mitigation actually in place:** explicit documentation that semantic search finds broad categories ("airfield," "river bridge," "urban edge"), while the change-detection engine is what surfaces small tactical-scale targets (e.g. a new 30m structure) — the two capabilities are complementary by design, and the UI/analyst documentation states this rather than implying semantic search alone can zero in on small hidden features.

---

## Query-Time Performance & Decision Policy

**Precomputed lookups only.** The change-detection pipeline is expensive and cannot run live per-query. A query resolves as two fast steps: a SQLite/R-Tree metadata lookup, and a FAISS vector search over the resulting candidate set, joined against precomputed change records.

**Metadata-filtered retrieval (§2.2.1) — how filtering actually works.** FAISS has no native attribute filtering; its only mechanism is an `IDSelector` passed via `SearchParameters(sel=...)`. A filter-native store (sqlite-vec, LanceDB, Qdrant) was evaluated as a replacement and **rejected**, because the documented FAISS weakness does not apply at this system's scale:
- The peer-reviewed finding (SIGMOD 2026, arXiv:2508.16263) is that Faiss-HNSW "struggles at 0.1% selectivity" — measured on corpora up to 10M vectors. The same study concludes that "with high selectivity (>50%), traditional methods like Faiss-HNSW are often more efficient."
- This system indexes an organiser-defined AOI: thousands to tens of thousands of tiles, not millions. Its realistic filters ("Sentinel-2, 2022–2025, this bbox") are **high**-selectivity — they pass a large fraction of the corpus, which is the regime where FAISS-HNSW is the better choice, not the worse one.
- Replacing a mature, widely-recognised index with a newer store to solve a failure mode outside our operating range would add risk without removing any.

**Implementation, three parts:**
1. **Pre-filter** — build the candidate ID allowlist in SQLite (date range, sensor) + R-Tree (bbox), then pass it to FAISS as an `IDSelectorBatch` (a hash set; the fast C++ path — custom Python selectors are documented as inefficient).
2. **Brute-force fallback below threshold** — when the allowlist is small (roughly a few thousand vectors), skip HNSW and scan those vectors exactly. At 512 dimensions this is sub-100ms and gives **exact** recall rather than approximate. At our corpus size this is the common path, not an edge case — which is precisely why no external store is needed.
3. **Correctness pinning** — use `bounded_queue=True` and pin a FAISS version carrying the fix from PR #5508; `IndexHNSW` previously **ignored** `SearchParameters.sel` when `bounded_queue=False`, silently returning IDs the filter should have excluded. Verify this behaviour in a test rather than assuming it.

*Documented migration trigger (FUTURE WORK):* if the corpus exceeds ~1M vectors, or filters routinely fall below 1% selectivity, migrate to LanceDB (embedded, IVF-based — IVF's non-monotonic search tolerates restrictive filters far better than HNSW). Not an MVP concern.

### Query-Time Visual Attribution (GAE Heatmaps)

After FAISS returns the top-K results for a text query, a **post-hoc attribution pass** runs on each result to localise *where within the 2.24km² crop* the match was triggered — addressing the single largest stated limitation of this architecture's retrieval scale.

**Method:** Generic Attention-model Explainability (Chefer, Gur & Wolf, ICCV 2021; `hila-chefer/Transformer-MM-Explainability`, MIT licence), applied to RemoteCLIP's ViT backbone. It combines gradient flow with attention weights across transformer layers to produce a per-patch relevance map — "Gradient-weighted Attention Rollout." The output is a spatial heatmap at ViT patch resolution: for a ViT-B/32 with 32×32-pixel patches over a 224×224 input, this is a 7×7 grid, where each cell covers roughly **320m × 320m at 10m imagery**. The analyst's search area drops from ~5 km² to ~0.1 km² per highlighted patch — a roughly **50× reduction**, rendered as a colour overlay ("attribution glow") on the retrieval tile in the UI.

**What this is not:** sub-metre localisation. Each patch is ~320m across, so the overlay says "look in this part of the tile," not "this specific building." State this honestly — a 50× search-area reduction is genuinely useful and demonstrable; "sub-hectare precision" would be an overclaim since the patches are larger than one hectare.

**Implementation:** one backward pass per query-result pair, at query time (not ingestion), only for the top-K results displayed. On GPU this adds tens of milliseconds per result; on CPU, low hundreds of milliseconds. No additional model — it reads the attention and gradient state of the same RemoteCLIP forward pass already happening. **Offline-safe:** no network calls, no new weights, uses the same packaged RemoteCLIP checkpoint.

**Provenance:** Chefer et al., *Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers*, ICCV 2021; code MIT-licensed at `github.com/hila-chefer/Transformer-MM-Explainability`.

Consequence of precomputation: a change can only be found once its background processing has completed — documented explicitly as a characteristic, not a promise of live analysis of brand-new imagery.

**Analyst-specified time windows (§2.2.2 — previously unaddressed).** The requirement is "for a specified area and time window," so an analyst asking specifically for *2022 vs 2025* must be served, even though ingestion only precomputes nearest-chronological (and optional same-season) pairs. Policy:
1. **If the requested pair was precomputed** → instant lookup, same as any other query.
2. **If not** → the system runs that single pair on demand, and **tells the analyst it is doing so**, with a progress indicator. This is seconds-to-minutes, not milliseconds — one pair through Phases 1–5. It is not a system fault; it is the honest cost of an arbitrary historical comparison, and the result is then cached into the catalog so the same request is instant thereafter.
3. **Latency reporting must distinguish these cases.** Quoting only the precomputed-lookup figure as "query latency" in the evaluation report would be misleading. Report precomputed lookup latency and on-demand pair latency as separate, clearly-labelled numbers.

**Semantic vs. change discrepancy policy** — a real scenario: RemoteCLIP retrieves a tile as a strong match for "newly built structures near a river," but the precomputed change record says "no significant change." Not a contradiction to arbitrate — two subsystems correctly answering two different questions. RemoteCLIP has no concept of time; it recognizes a tile matches "a structure near a river" as a single snapshot, with no way to verify "newly," which only the change-detection layer can check.
1. **Never merged into one score.** Semantic match strength and change status are always shown as two distinct, separately labeled pieces of information.
2. **Temporal query language boosts, never hides.** Queries containing "newly," "recently," etc. rank results with a confirmed matching change record higher — a semantic match without one is never removed, since "this looks like what you described, though records show no recent change here" is still useful.
3. **Missing change data is a distinct state from "no change."** If background processing hasn't completed for a location, it's shown as "Analysis Pending," never conflated with a confirmed "no change" result.

---

## Known Resolution Limits (state explicitly in the submission, not as a hidden gap)
- 10m is the ceiling across every permitted dataset source (Sentinel-2, Landsat, open Bhuvan products) — applies to every team in the evaluation equally.
- Individual vehicles are sub-pixel and cannot be reliably resolved. "Vehicle concentration" queries are interpreted at the texture/spatial-pattern level, not as individual object detection.
- **Road development is not claimed as a verified capability** — reported as low-confidence "possible linear feature" only (see Phase 5 for the published evidence behind this scope decision).
- **Not yet implemented:** feedback-driven reranking from confirm/reject history.
- View-angle differences and terrain shadow: addressed via DEM terrain-risk masking and TASC, not eliminated — steep terrain carries reduced confidence by design, a graded mitigation rather than a claim of full resolution.
- SAR fallback (cloud-penetrating) is deferred/stretch scope, not available by default.

## Evaluation Instrumentation (§2.3 — must be built in from the first commit)

The submission must report "indexed area, number of scenes or tiles, build time, storage footprint, query latency and hardware used." **Build time and query latency cannot be reconstructed retroactively** — if this is not instrumented from the start, the only way to produce the numbers at the end is re-running full ingests purely to measure them. Instrument now:
- **Build time** — wall-clock, logged per stage (ingest, capability detection, masking, COG conversion, tiling, embedding, indexing, and each change-detection phase), timestamped to the catalog.
- **Storage footprint** — filesystem stat over COGs, the SQLite file (catalog + vectors), and any cached intermediates, recorded per ingest.
- **Indexed area** — sum of scene footprints, deduplicated for overlap, computed from the R-Tree geometries.
- **Tile counts** — from window/tile enumeration at tiling time.
- **Query latency** — p50/p95/p99 distributions, recorded separately for (a) precomputed semantic lookup, (b) metadata-filtered semantic lookup tagged with filter selectivity, and (c) on-demand arbitrary-pair change analysis.
- **Hardware** — CPU, RAM, GPU, OS captured once per run.

Emit all of this as a machine-readable JSON manifest alongside dataset checksums and version pins, so the evaluation report is reproducible rather than hand-assembled.

---

## Model & Data Provenance and Licensing (§2.2.7 — mandatory declaration)

The constraint is explicit: pretrained public models may be used "provided that their origin and licence are declared and the required weights are packaged for offline use."

| Artifact | Origin | Licence | Notes |
|---|---|---|---|
| **RemoteCLIP** (code) | Liu, Chen et al., *RemoteCLIP: A Vision Language Foundation Model for Remote Sensing*, IEEE TGRS 2024 (arXiv:2306.11029); repo `ChenDelong1999/RemoteCLIP` | **Apache-2.0** (repo LICENSE) | |
| **RemoteCLIP** (weights) | HuggingFace `chendelong/RemoteCLIP` (RN50 / ViT-B-32 / ViT-L-14, OpenCLIP format) | **UNSTATED — model card is empty, no `license:` field** | **Declare this honestly as unstated rather than assuming Apache-2.0 extends to the weights.** Highest-attention compliance item in this project. |
| **OpenCLIP backbone** | `mlfoundations/open_clip` | **MIT** | Permissive |
| **OpenAI CLIP** (upstream) | `openai/CLIP` | **MIT** for code; weights licence not separately stated (see repo issue #203) | |
| **RS caption training data** | RSICD; RSITMD (`AICyberTeam/AMFMN`); UCM-Captions | **No licence file / research-only.** UCM imagery itself is USGS public domain, but the dataset page states research purposes. RSICD imagery scraped from Google Earth/Baidu/MapABC/Tianditu → upstream-terms risk. | Restrictions attach to the *data*; whether they bind derived *weights* is a legal grey area |
| **DOTA** (in the DET-10 pretraining chain) | captain-whu.github.io/DOTA | **"All images and their associated annotations in DOTA can be used for academic purposes only, but any commercial use is prohibited"** | Most restrictive link in the chain. Fine for an academic hackathon submission; would need legal review before any commercial distribution. |
| **s2cloudless** | `sentinel-hub/sentinel2-cloud-detector` | **CC-BY-SA-4.0** (share-alike applies to modifications of the model) | **Demoted off the MVP primary path.** Trained on L1C and documented not to perform well on L2A, which is what we ingest. SCL is the sole primary mask; s2cloudless is retained only as an optional secondary check for L1C input. If not shipped, this licence does not apply to the deliverable at all. |
| **CartoDEM** | ISRO/NRSC, Cartosat-1; via Bhuvan/Bhoonidhi | **NDSAP → Government Open Data License – India (GODL-India)** | ~10m posting (1/3 arc-sec); NRSC states 8m LE90 vertical, 15m CE90 planimetric accuracy. **Download quota: 20 tiles/day** per Bhuvan FAQ — plan staging accordingly, this can bottleneck preparation. |
| **Sentinel-2** | Copernicus / ESA | **Free, full and open** (EU Reg. 1159/2013) | Attribution mandatory: "Copernicus Sentinel data [Year]", or "modified Copernicus Sentinel data [Year]" for derivatives |
| **Landsat Collection 2** | USGS | **U.S. Public Domain** — "Permission is not required for use" | Acknowledgment requested, not required |
| **OSCD** | Daudt et al., IGARSS 2018 (DOI 10.1109/IGARSS.2018.8518015); IEEE DataPort DOI 10.21227/asqe-7s69 | Research/benchmark — no blanket open-commercial licence | Cite the IGARSS paper; treat as research-only |

**Offline weight packaging procedure:**
- Pre-download all weights at build time (`huggingface_hub.snapshot_download`) into a bundled `HF_HOME`; set **`HF_HUB_OFFLINE=1`** at runtime so no HTTP calls are made to the Hub — this also suppresses the version-check call that fires even on cached files.
- **Preferred:** load the RemoteCLIP `.pt` checkpoint by explicit local path via OpenCLIP, bypassing Hub calls entirely. Some libraries call `model_info()` (a network call) *before* consulting cache and fail outright under `HF_HUB_OFFLINE=1` in air-gapped environments — so test the genuine offline path rather than assuming a warm cache is sufficient.
- **SHA-256 checksum every weight and dataset artifact**, recorded in the provenance manifest, and pin exact versions in a lockfile (rasterio clear of the 1.4.2 windowed-read leak; sqlite-vec; UMAP/HDBSCAN).

---

## Differentiation Strategy — Top 5 Demonstration Moments

These are not features added for novelty — they are capabilities already in the architecture, framed for maximum impact during a live demonstration. Listed in the order they should be shown.

| # | Name | What the judge sees | Why it works |
|---|---|---|---|
| **1** | **Ablation Transparency (live toggle)** | Suppression stack is toggled off live; the map floods with false positives; toggled back on, they disappear. | Every other team *claims* false-alarm reduction. This *shows* it, live, unfakeably. The ablation table is already a mandatory evaluation artifact — this makes it visible in-app. |
| **2** | **Negative-Evidence Confirmation** | An analyst queries an area and gets "this area was observed on [dates] with [X]% coverage — no change exceeding your threshold was detected." Exportable with full provenance. | "We confirmed nothing was built there" is an intelligence product. "No results" is not. Possible only because change detection precomputes for every overlapping pair, not on demand. |
| **3** | **Decision Trace (per detection)** | Drill into any detection: masking path, alignment method + RMSE, PIF tier, four confidence terms, direction classification with its own confidence. | Every detection carries its own reproducible chain of evidence. The word *reproducible* matters for defence — it means a superior can independently verify an analyst's conclusion. |
| **4** | **GAE Attribution Heatmap** | A search result tile lights up with a colour overlay showing *which part* of the 2.24km² crop triggered the match, reducing the analyst's search area ~50×. | Directly demonstrates that the system doesn't just find the right tile — it points within it. No other team at this scale will have sub-tile localisation from a retrieval model. |
| **5** | **Temporal Coverage Heatmap** | Shown first, before any query — a choropleth of how well-observed each part of the AOI is, derived from catalog cloud-trust scores. | Frames every subsequent result: the judge already knows where the system's evidence is strong and where it's thin, making every answer more credible. |

**Pipeline Visibility Panel** (not in the top 5 but shown throughout): the right sidebar displays live stage-by-stage progress during ingestion, so judges see the architecture executing rather than trusting a diagram. This is a demo-quality feature, not a gimmick — it shows the capability-detection branching, the masking path chosen, and the tile/embedding counts advancing in real time.

---

## Architecture Decision Log — v2.2 freeze (+ v2.2.1 critical fixes, + v2.2.2 acceptance-review fixes, + v2.2.3 implementation-gate fixes, + v2.2.4 / v2.2.5 red-team fixes, + v2.2.6 freeze-gate, + v2.2.7 innovation layer)

Decisions taken under a "strengthen, do not redesign" review standard: a component is replaced only if the current design demonstrably fails, the replacement fixes that failure, and it adds no complexity or offline-constraint violation. Optimizations and alternatives are declined.

| Decision | Status | Rationale |
|---|---|---|
| Change-direction classification (Phase 4b) | **LOCK** | §2.2.2 requires appearance/disappearance/expansion/contraction; a magnitude-only mask cannot answer it. Training-free, no new dependency. |
| Band specification for differencing/PCA | **LOCK** | Genuine ambiguity and error risk — Celik's PCAKM operates on a single difference image, not a 13-band stack. |
| Analyst-specified time-window policy | **LOCK** | §2.2.2 requires it; previously unaddressed. Policy + caching, no new technology. |
| Model/data provenance & licensing | **LOCK** | §2.2.7 mandates declaration. Previously absent. |
| Evaluation instrumentation | **LOCK** | §2.3 deliverable; build time and query latency cannot be reconstructed retroactively. |
| Tile-boundary halo + best-match merge | **LOCK** | Systematic blind spot along every seam. A parameter and a merge rule — not a component change. |
| Export format + provenance block (GeoJSON) | **LOCK** | §2.2.5 required it; previously asserted rather than specified. |
| Road detection scoped to "possible linear feature" | **LOCK** | Removes an indefensible claim. Negative complexity. |
| s2cloudless demoted off primary path | **LOCK** | L1C-trained, pipeline is L2A. Removes a dependency and simplifies the critical path. |
| Metadata-filtered retrieval | **MODIFY** | Real §2.2.1 gap, fixed in place (pre-filter → `IDSelectorBatch`, brute-force fallback, version pinning) rather than by replacing the index. |
| Archive-wide clustering | **MODIFY** | Requirement genuine; implemented with explainable K-means rather than UMAP→HDBSCAN. |
| FAISS → sqlite-vec / LanceDB / Qdrant | **REJECT** | The cited FAISS-HNSW filtering weakness occurs at ≤0.1% selectivity on multi-million-vector corpora. Our corpus is thousands to tens of thousands of tiles with high-selectivity filters — the regime where FAISS-HNSW is documented as the *better* choice. An alternative, not a fix. |
| UMAP→HDBSCAN clustering | **FUTURE WORK** | Better outlier handling, at the cost of explainability and two heavy dependencies. |
| LanceDB migration | **FUTURE WORK** | Trigger: corpus >1M vectors, or filters routinely below 1% selectivity. |
| Sentinel-1 SAR pipeline | **FUTURE WORK** | Isolated, analyst-override only. |
| Feedback-driven reranking | **FUTURE WORK** | Confirm/reject history captured now; reranking not implemented. |
| **v2.2.1 — NaN-safe numerical contract** | **LOCK** | Masking produces NaN; ECC, PCA and KMeans all raise on it. Crash-level, not accuracy-level. |
| **v2.2.1 — SQLite WAL + connection-per-process + `BEGIN IMMEDIATE`** | **LOCK** | Default journal mode throws `SQLITE_BUSY` under our reader/writer topology; a connection inherited across `fork()` can corrupt the DB silently. |
| **v2.2.1 — Grid Check hardened to CRS + origin + transform + resolution** | **LOCK** | Same-CRS does not imply same grid; sub-pixel origin offset produces spurious edge change that co-registration cannot fix. |
| **v2.2.1 — K-Means cluster-label resolution by centroid magnitude** | **LOCK** | Cluster IDs are arbitrary and silently invert between scenes; reading change from the index produces inverted results. |
| **v2.2.1 — Job state machine (`queued → processing → completed`) + staging/atomic rename + startup sweep** | **LOCK** | Interruption mid-job otherwise leaves orphaned files and permanently stuck jobs. |
| **v2.2.2 — NaN gate on the embedding path (M-1)** | **LOCK** | A ViT propagates NaN *silently*; a NaN vector in the FAISS graph corrupts traversal and returns garbage rankings with no error raised anywhere. Worst failure class in the design — wrong results, no signal. |
| **v2.2.2 — Same-sensor constraint on change-detection pairing (M-2)** | **LOCK** | Footprint overlap alone permitted Sentinel-2↔Landsat differencing. Without HLS-style BRDF + bandpass harmonisation, sensor difference presents as change — undermining the entire false-alarm stack from upstream. Restricts differencing only; cross-sensor retrieval and display unaffected. |
| **v2.2.2 — `NormClusterDist` specified as per-scene percentile rank (M-3)** | **LOCK** | Term was unbounded and scene-dependent, so not implementable as written; any fixed normalisation makes confidence non-comparable across scenes and breaks review-queue ranking. |
| **v2.2.3 — Morphology corrected to opening + closing (4 ops), with linear-branch ordering constraint** | **LOCK** | Document claimed noise removal *and* hole filling from erosion→dilation, which is an opening and does only the former. Would have silently delivered half the stated behaviour. Opening also severs thin features, so the linear-feature path must run pre-opening. |
| **v2.2.7 — GAE attribution heatmaps adopted** | **LOCK** | Solves the 2.24km tile-size limitation at query time. Chefer ICCV 2021, MIT, confirmed on ViT-B/32. |
| **v2.2.7 — MGRS geocoding, Evidence Card, Pipeline Visibility adopted** | **LOCK** | Low-effort defence-workflow features built on existing data. |
| **v2.2.7 — GAE visual attribution heatmaps at query time** | **LOCK** | Addresses the largest stated limitation (2.24km² tile dilutes small features) without changing the embedding model or tile size. Post-hoc on existing forward pass, MIT-licensed, no new weights. |
| **v2.2.7 — MGRS geocoding in catalog and all exports** | **LOCK** | One function call per coordinate; demonstrates the system was built for Indian Army operational context (MGRS is the NATO/Indian Army standard), not adapted from a civilian tutorial. |
| **v2.2.7 — Evidence Card (HTML summary per detection)** | **LOCK** | Formatted view of existing provenance data, not new computation. Produces a standardised intelligence summary passable up a command chain. |
| **v2.2.7 — Pipeline Visibility Panel** | **LOCK** | Right sidebar shows live stage-by-stage progress with branching paths visible. Demo-quality and analyst-trust feature. |
| **v2.2.6 — Connected-component sets named and separated: Classification Components (Phase 4b) vs Change Blobs (Phase 5), with fixed execution order** | **LOCK** | Two different CC operations were conflated and their order was contradictory — three plausible implementations existed, producing different results. Morphology now explicitly never touches Classification Components. |
| **v2.2.6 — `NDVI_DROP_SIG` named; Review Queue roadmap wording corrected** | **LOCK** | Last unnamed threshold in the document; and "not yet implemented" conflicted with the instrument-from-first-commit requirement for evaluation reporting. |
| **v2.2.5 — `TerrainFlatness` functional form specified as cos(slope), floored at 60°** | **LOCK** | Previously described as intent ("reduced confidence") with no formula — not implementable, and invites the same "show me the function" question the `rho` threshold correctly anticipates elsewhere. |
| **v2.2.5 — Ablation study made a mandatory evaluation requirement** | **LOCK** | Full-pipeline precision/recall proves the system works but *not* that Phases 1–3 contribute anything — which is this submission's central claim. Only measurement answers it. |
| **v2.2.5 — Phase correlation recorded as evidenced optional alternative to ECC** | **ACCEPTED (optional)** | Legitimately better-evidenced primitive for sub-pixel co-registration, with a natural per-block confidence measure. Recorded on technical merit; adopted only on measured accuracy. ECC+ORB remains the specified path. |
| **Replacing ECC/ORB because the problem statement "excludes" them** | **REJECT** | No such exclusion exists in PS 26227. §2.2.3 requires "quality masks, normalization, confidence estimates or equivalent mechanisms" — there is no prohibited-technique list. The constraint originated in an external review's own assumption, restated across rounds as established fact. Replacing a sound, failure-handled subsystem to satisfy a non-existent rule is rejected. |
| **v2.2.4 — Phase 2 reordered: Grid Check before Mutual Validity Mask** | **LOCK** | ANDing two masks not yet on an identical pixel grid combines values from different ground positions — silently wrong, not erroneous. Impossible execution order as previously written. |
| **v2.2.4 — ECC `rho` correlation-coefficient gate** | **LOCK** | Exception handling covered only the loud failures. ECC can converge to a bad local optimum and return a transform with no exception, poisoning every downstream decision for that pair. Low-texture regions (water, bare fields) are the common trigger. |
| **v2.2.4 — Unsourced "83%" runaway-merge statistic removed** | **LOCK** | Specific statistic carried over without a citation. Mechanism retained and explained; the number is gone rather than presented unsourced to a technical panel. |
| **v2.2.4 — GPU/compute assumption and embedding-throughput reporting stated** | **LOCK** | Hardware was implicit; §2.3 requires reporting hardware used, and on CPU-only hardware embedding is the ingestion bottleneck. |
| **v2.2.3 — Index thresholds given distinct names; B11 20m→10m bilinear resampling rule specified** | **LOCK** | A single undefined identifier `threshold` was used for two different comparisons — literally not codeable. B11/B08 resolution mismatch had no stated handling; nearest-neighbour upsampling would manufacture false NDBI change along every edge. |

---

## Differentiation Features (Innovation Layer)

Capabilities beyond baseline requirements that competing teams are unlikely to implement. Each is built on the existing architecture without modifying the core pipeline.

| Feature | What it does | Why it differentiates | Effort |
|---|---|---|---|
| **Ablation Transparency (live toggle)** | UI toggle disabling suppression stages, showing the false-alarm flood in real time | Most teams won't run ablation at all; this lets judges *see* the alternative | Medium |
| **Negative-Evidence Confirmation** | Exports provenance-backed confirmation of *absence* — "observed on [dates], [X]% coverage, no change found" | Requires precomputed analysis even for negative results (ingest-time pipeline) | Low |
| **GAE Attribution Heatmaps** | Per-result patch-level localisation within the 2.24km retrieval tile (~50× search-area reduction) | Requires ViT attention access + Chefer's rollout; most teams treat the model as a black box | Low-Medium |
| **MGRS Geocoding** | Every tile, detection and export carries a 10-digit Military Grid Reference | One call per coordinate; signals defence-workflow design intent | Low |
| **Visual Evidence Card** | One-click printable intelligence summary per confirmed detection | No search-engine project will produce a printable intelligence product | Low |
| **Decision Trace** | Per-detection full processing chain exportable for independent verification | Already designed — a *framing* of existing provenance as accountability | Zero |
| **Confidence Decomposition View** | Segmented bar showing which factor drove the score down | Four terms already computed — UI rendering only | Low |
| **Temporal Coverage Heatmap** | Pre-query observation-density map showing where the system is strong and where it's thin | Derived from existing catalog metadata; no team treats their catalog as an analytical product | Low |
| **Pipeline Visibility** | Live ingestion progress showing each processing stage with counts | Reads from the existing job-state table | Low |

**Deferred (document, don't build):** IR-MAD replacing PCAKM (build PCAKM first, measure, swap if measured improvement); phase-correlation alignment (adopt on measured accuracy); analyst watchlist with auto-alerting (after review queue works).

---

## Calibration & Validation Strategy (required before submission, not optional polish)

Every numeric threshold in this document — the AOI cloud trust curve, the 0.25 whiteness threshold, the 0.5px RMSE trigger, PCA variance retention, K, the confidence formula's weights, the mutual-coverage floor, the morphological structuring-element size and minimum mapping unit, and the named index thresholds (`NDBI_BUILT_MIN`, `NDVI_BUILT_MAX`, `MNDWI_WATER_MIN`, `NDVI_VEG_MIN`, `NDVI_DROP_SIG`, plus the shape-descriptor cutoffs) — is a reasonable domain-standard starting point, not a validated value. Index thresholds in particular are strongly scene-dependent and should be derived per-scene (Otsu) rather than transferred as fixed values.

**Two-stage validation:**
1. **OSCD (Onera Satellite Change Detection)** — 24 Sentinel-2 city pairs (2015–2018), pixel-level ground truth for 14 pairs, urban change focus, free via IEEE Dataport. Run the full pipeline against it, measure actual precision/recall/F1, and set every threshold above from real results.
2. **Local re-verification** — once calibrated on OSCD, re-check against a real Indian target landscape: **Noida International Airport (Jewar, UP)** — agricultural land (2018–2019) → major construction (2023–2024) — a visually unambiguous, large-scale end-to-end sanity check that the OSCD-tuned thresholds still behave sensibly on local terrain and land-use patterns.

**Ablation is mandatory, not optional — it is the only thing that proves the suppression stack does anything.** Reporting precision/recall for the full pipeline shows the system works; it does *not* show that the false-alarm suppression machinery (Phases 1–3) contributes anything, which is the central claim of this submission. Required measurements, same imagery, same thresholds:

| Configuration | What it isolates |
|---|---|
| Full pipeline | Headline precision/recall/F1 |
| Suppression stack **off** (raw difference → PCA/K-Means → cleanup only) | Total contribution of Phases 1–3 |
| Quality masking off (SCL / cloud trust disabled) | Contribution of masking alone |
| Radiometric normalisation off (TASC + PIF disabled) | Contribution of normalisation alone |
| Co-registration off (difference on the reprojected grid only) | Contribution of alignment alone |

Report each as precision/recall/F1 plus raw false-positive count. **If disabling a stage does not measurably degrade results, that stage is not earning its place** — say so plainly rather than retaining it for appearance. This ablation is the direct answer to the hardest fair question a reviewer can ask: "how do you know your false-alarm suppression actually suppresses false alarms?" That question cannot be answered by design reasoning, only by measurement.

**Supplementary labelled set.** Alongside OSCD, hand-annotate a small set of Sentinel-2 pairs over Indian terrain, labelling real change *and*, where a detection is wrong, the false-positive source (cloud / shadow / seasonal / illumination / misregistration). Per-source labelling is what lets the ablation table attribute improvements to specific stages. State results as "validated on N manually-labelled pairs covering X km²" — small honest numbers beat no numbers.

This is the step that turns "architecturally sound on paper" into "actually proven" — no amount of further design review substitutes for it.
