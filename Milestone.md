# IRIS Project Milestones & Execution Log

**System Designation:** IRIS (Intelligent Retrieval & Interpretation System)  
**Problem Statement ID:** 26227 — *Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery*  
**Issuing Authority:** Ministry of Defence / Indian Army (DGIS)  

This document chronicles the six implementation milestones achieved during the engineering of IRIS, detailing the capabilities built, technical decisions executed, and deliverables verified.

---

## Milestone 1 — Universal Ingestion & On-Screen Imagery Visualization
**Core Objective:** Establish the offline geospatial data ingestion foundation, convert diverse raw satellite products into Cloud-Optimized GeoTIFFs (COGs), and render raster imagery seamlessly on an interactive map.

### Accomplishments
1. **Universal Format Loader (`backend/ingestion/loader.py`):**
   * Engineered multi-format parser supporting Sentinel-2 (.SAFE directories and JPEG 2000 `.jp2` files), Landsat Collection 2 Level-2 products, and standard GeoTIFFs.
   * Built automated capability detection to audit band configurations, native Coordinate Reference Systems (CRS), and spatial resolutions before processing.
2. **Native Quality Masking & COG Conversion (`backend/ingestion/masking.py`, `backend/ingestion/cog.py`):**
   * Extracted native categorical quality bands (Sentinel-2 Scene Classification Layer / SCL, Landsat QA_PIXEL) to mask clouds, shadows, and snow before reprojection.
   * Synthesized true-color RGB composites and converted them into internally tiled, over-viewed Cloud-Optimized GeoTIFFs (DEFLATE compression, 512×512 block size).
3. **SQLite Spatial Catalog (`backend/catalog/database.py`):**
   * Initialized relational database with Write-Ahead Logging (`PRAGMA journal_mode=WAL`), `PRAGMA synchronous=NORMAL`, and R-Tree spatial indexing for bounding box lookups.
   * Computed 10-digit Military Grid Reference System (MGRS) coordinates for all ingested scenes.
4. **Local Tile Server & Map Presentation (`backend/main.py`, `frontend/src/components/MapView.jsx`):**
   * Integrated TiTiler dynamic tile server mounted directly on `127.0.0.1:8000`.
   * Built MapLibre GL frontend displaying local raster XYZ tiles with smooth zooming, panning, and extent-fitting controls without any external web basemap.

---

## Milestone 2 — Semantic Retrieval & Foundation Model Vector Indexing
**Core Objective:** Enable natural language semantic search and image-to-image similarity retrieval across satellite archives using deep vision-language representations.

### Accomplishments
1. **Uniform Grid Tiling & Valid-Pixel Gating (`backend/embedding/tiler.py`):**
   * Implemented fixed 224×224px surface tiling with 30% stride overlap, matching the RemoteCLIP receptive field.
   * Enforced minimum valid-pixel thresholding (discarding tiles with <60% valid ground pixels) and neutral mean imputation to prevent NaN corruption of vector spaces.
2. **RemoteCLIP Vision-Language Inference (`backend/embedding/embedder.py`):**
   * Integrated RemoteCLIP (ViT-B/32 backbone, 512-dimensional output) pre-trained specifically on remote sensing aerial and orbital imagery.
   * Packaged weights locally for air-gapped CPU/CUDA execution, generating L2-normalized embeddings.
3. **FAISS Dynamic HNSW Vector Index (`backend/embedding/index.py`):**
   * Configured Hierarchical Navigable Small World (`IndexHNSWFlat`) vector graph supporting dynamic incremental vector addition without index rebuilds.
   * Linked vector indices with SQLite metadata records for fast hybrid spatial-semantic filtering.
4. **Search Interface & Interactive Navigation (`frontend/src/components/SearchBar.jsx`):**
   * Built natural language search bar with category presets and top-$K$ results drawer.
   * Implemented interactive "fly-to" coordinate navigation and visual boundary highlight upon result selection.

---

## Milestone 3 — Explainable Multi-Temporal Change Detection Pipeline
**Core Objective:** Implement a 5-phase physical change detection pipeline that identifies genuine physical transformations while suppressing false alarms from clouds, sun angles, and registration jitter.

### Accomplishments
1. **Phase 1 — Quality Masking & Trust Scoring:**
   * Computed mutual validity mask via logical AND of observation pairs; scored per-AOI cloud trust ($1 - \text{cloud\_fraction}$) to scale confidence.
2. **Phase 2 — Geometric Alignment & Co-Registration:**
   * Gated processing with rigid 4-point grid equality checks (CRS, origin, transform, resolution).
   * Applied sub-pixel enhanced correlation coefficient (`cv2.findTransformECC`) with correlation threshold ($\rho \ge 0.8$) and ORB+RANSAC fallback.
   * Integrated CartoDEM 30m digital elevation model to compute terrain slope and apply a cosine-based terrain-flatness confidence discount.
3. **Phase 3 — Radiometric Normalization:**
   * Computed Topographic and Sun-Angle Correction (TASC) using product solar zenith/azimuth metadata.
   * Identified Pseudo-Invariant Features (PIFs) via joint temporal stability and registration confidence to fit linear brightness normalizations.
4. **Phase 4 & 4b — Statistical Differencing & Direction Classification:**
   * Decomposed NIR difference channels via local block-PCA and clustered via K-Means ($K=2$).
   * Built classification component matching (IoU $\ge 0.5$) and signed index deltas ($\Delta\text{NDBI}$, $\Delta\text{NDVI}$, $\Delta\text{MNDWI}$) to categorize changes into **Appearance**, **Disappearance**, **Expansion**, or **Contraction**.
5. **Phase 5 — Morphological Cleanup & Seasonal Filtering:**
   * Applied 4-step morphological opening (noise removal) and closing (gap filling) on change blobs.
   * Queried catalog historical baseline across same-calendar-month observations to discard cyclical agricultural/phenological variations.
   * Computed 4-term normalized confidence score per candidate polygon.
6. **Comparison UI (`frontend/src/components/ChangeComparison.jsx`):**
   * Created interactive before/after split-screen swipe comparison slider and polygon overlay.

---

## Milestone 4 — Analyst Review Queue, Watchlist, & Provenance Export
**Core Objective:** Establish verification and auditability workflows for intelligence analysts, including review queues, persistent monitoring, and defence-compliant export dossiers.

### Accomplishments
1. **Review Queue & Decision State Machine (`frontend/src/components/ChangeResults.jsx`):**
   * Interactive triage interface allowing analysts to Confirm, Reject, Flag, or Monitor detected change candidates.
   * Multi-faceted filtering by confidence threshold, change type, direction, and scene pairs.
2. **Watchlist Monitoring Subsystem (`frontend/src/useWatchlist.js`):**
   * Pinpoint critical Areas of Interest (AOIs) with automatic 500m geofence radius.
   * Evaluates incoming temporal pairs against pinned sites and fires UI notifications upon new changes.
3. **GeoJSON Provenance Export (`backend/api/export.py`):**
   * Single-click export of validated changes in RFC 7946 GeoJSON format, embedding full operational provenance (source scene IDs, acquisition dates, sensor calibrations, alignment RMSE, confidence decomposition, operator sign-off).
4. **Visual Evidence Card Dossiers (`backend/api/evidence.py`):**
   * Formatted printable single-page intelligence briefs featuring before/after image chips, MGRS coordinates, four-term confidence breakdowns, and SHA-256 raster integrity checksums.

---

## Milestone 5 — Full Air-Gapped Offline Hardening & System Documentation
**Core Objective:** Eliminate all external network dependencies, integrate native operating system file dialogs, and formalize the complete technical documentation.

### Accomplishments
1. **100% Offline Air-Gap Verification:**
   * Audited all frontend assets; removed remote CDN links (unpkg, external stylesheets, remote fonts).
   * Bundled all fonts, icons, and JavaScript libraries locally within the project.
   * Verified zero HTTP/HTTPS network calls to any host other than `127.0.0.1`.
2. **Native Windows Explorer Dialogs (`backend/api/dialog.py`, `frontend/electron/main.js`):**
   * Integrated native Windows `FolderBrowserDialog` and `OpenFileDialog` via Electron IPC and background PowerShell interop.
   * Allowed seamless 1-click importing of external folders (.SAFE datasets) and files directly through native Windows Explorer windows.
3. **Comprehensive System Documentation (`documentation.md`):**
   * Authored full 13-part master documentation manual detailing system design, mathematical formulas, API specifications, and operational playbooks.

---

## Milestone 6 — Standalone Desktop Executable (`.exe`), 1-Click Launcher, & Brand Identity
**Core Objective:** Package IRIS into a zero-configuration standalone Windows desktop software executable, provide 1-click desktop launching, and establish brand identity.

### Accomplishments
1. **Production Electron Compilation & Packaging (`frontend/package.json`):**
   * Configured `electron-builder` with custom distribution targets (`release/win-unpacked/IRIS.exe` and `release/IRIS Setup 0.1.0.exe`).
   * Packaged production-minified React/Tailwind/MapLibre bundle directly into `resources/app.asar`.
2. **1-Click Windows Desktop Launcher (`Launch_IRIS.bat`, `launch.ps1`):**
   * Engineered intelligent launcher that checks whether the Python AI backend on `127.0.0.1:8000` is active; starts it silently in the background if closed.
   * Automatically launches the native **`IRIS.exe`** desktop window without requiring terminal commands or web browsers.
   * Placed direct **`IRIS`** shortcut on the Windows Desktop (`Desktop/IRIS.lnk`).
3. **Custom Insignia & Brand Packaging:**
   * Converted the user's custom monochrome satellite-eye insignia into multi-resolution `.ico` and `.png` assets (16×16 to 512×512).
   * Embedded the custom icon into the Windows executable, installer, Windows taskbar, window titlebar, and desktop shortcut.
   * Integrated the circular emblem into the application toolbar header.
4. **Header Refinement:**
   * Removed redundant status text and badges for a clean, distraction-free intelligence workspace.

---

*IRIS — Successfully Delivered through 6 Engineering Milestones.*
