# IRIS: Intelligent Retrieval & Imagery Surveillance

> **Offline Desktop System for Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery**  
> *Developed for Smart India Hackathon (SIH) — Problem Statement ID: 26227*  
> *Proponent Organization: Ministry of Defence / Indian Army (DGIS)*

---

## 1. Problem Statement

* **Problem Statement ID:** `26227`
* **Title:** Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
* **Category:** Software / National Security / Geospatial Intelligence
* **Proponent:** Ministry of Defence / Indian Army (DGIS)

### Background & Operational Challenge
Earth-observation archives are expanding exponentially with multi-temporal, multi-spectral, and multi-sensor acquisitions from constellations such as Sentinel, Landsat, and Bhuvan. Conventional military and defense geospatial catalogues are indexed strictly by static metadata—coordinates, bounding boxes, acquisition dates, sensor models, and product levels. While effective when the analyst already knows *where* and *when* an event occurred, analysts cannot query archives by **semantic meaning** (e.g., *"new forward airstrip construction in arid terrain"* or *"floating pontoon bridge deployment along river corridor"*).

Furthermore, multi-temporal change detection algorithms in existing GIS tools produce intolerable rates of false alarms caused by cloud edges, seasonal phenology cycles, sun-angle shifts, and sub-pixel geometric registration jitter. Translating recent foundation-model breakthroughs into a reliable operational system requires:
1. Complete, air-gapped **offline execution** on portable field workstations.
2. An **explainable, non-black-box change engine** satisfying defence accountability standards.
3. True **incremental ingestion** without full index recomputations.
4. Rigorous preservation of **geospatial and analytical provenance**.

---

## 2. Project Overview

**IRIS (Intelligent Retrieval & Imagery Surveillance)** is a standalone, offline desktop application engineered to turn vast, unlabelled satellite imagery archives into an intuitively searchable, explainable intelligence platform.

### Core Capabilities
* **Natural-Language Semantic Retrieval:** Free-text natural language queries powered by a domain-specialized RemoteCLIP vision-language model, searching the archive by human meaning without manual annotation.
* **Image-to-Image Visual Similarity:** "Find Similar Sites" allows an analyst to select any land-use pattern (e.g., military encampment, radar site, deforestation patch) and instantly discover matching sites across all indexed dates and footprints.
* **5-Phase Explainable Change Engine:** Replaces black-box deep learning differencing with an auditable physical pipeline: native categorical quality masking, sub-pixel grid co-registration, radiometric normalization (TASC & PIF), block-PCA/K-Means difference clustering, and multi-spectral index directional classification.
* **Directional Change Classification:** Automatically classifies candidate features into physical categories: **Appearance**, **Disappearance**, **Expansion**, or **Contraction**, coupled with spectral index deltas (ΔNDBI, ΔNDVI, ΔMNDWI).
* **Analyst Review Queue & Provenance Audit Trail:** Interactive validation UI for intelligence officers to verify, reject, flag, or monitor changes with one-click **GeoJSON audit exports** and printable **Visual Evidence Dossiers**.
* **100% Offline Desktop Shell:** Zero external network calls. Bundled as a native Windows desktop application with native OS file dialogs and an embedded local tile server.

---

## 3. Solution Summary

| Subsystem | Technology | Purpose & Architectural Rationale |
|---|---|---|
| **Desktop Shell** | **Electron** | Native Windows desktop application with sandboxed IPC and native Windows Explorer file/folder picker dialogs. |
| **User Interface** | **React + TailwindCSS + MapLibre GL** | QGIS-inspired intelligence console with real-time vector tile rendering and split-screen before/after comparison slider. |
| **Application Server** | **FastAPI (Python)** | High-performance asynchronous backend bound exclusively to `127.0.0.1:8000`. |
| **Raster Engine** | **GDAL / Rasterio / TiTiler** | Memory-bounded streaming raster windowing and on-demand local XYZ tile serving from Cloud-Optimized GeoTIFFs (COGs). |
| **Embedding Model** | **RemoteCLIP (ViT-B/32)** | Domain-adapted satellite foundation model mapping 224×224px surface crops into a 512-dimensional joint semantic space. |
| **Vector Engine** | **FAISS (HNSW)** | Hierarchical Navigable Small World vector graph enabling sub-100ms similarity lookups with dynamic vector insertion. |
| **Metadata & Spatial Catalog** | **SQLite (WAL + R-Tree)** | Embedded, serverless relational catalog with R-Tree spatial indexing, Write-Ahead Logging concurrency, and MGRS grid coordinates. |

---

## 4. Architecture Note

> 📄 **Complete Technical Document:** Read the full [Architecture Note & Systems Design Specification](architecture.md).

### Architecture Requirements Summary
1. **Air-Gapped Operational Sovereignty:** Hard constraint: zero network traffic permitted outside `127.0.0.1`. All foundation model weights, tile services, spatial lookups, and Python wheels run self-contained locally.
2. **Explainable Non-Black-Box Pipeline:** Deep learning end-to-end change detection models hallucinate and fail defence verification standards. IRIS computes an explainable 4-term confidence score:
   $$\text{Confidence} = 0.35 \times \text{NormRMSE} + 0.35 \times \text{NormClusterDist} + 0.15 \times \text{TerrainFlatness} + 0.15 \times \text{ValidCoverage}$$
3. **Memory-Bounded Streaming Processing:** Raw 100 km × 100 km Sentinel-2 tiles unpack to gigabytes in memory. IRIS enforces overlapping strip-window processing (16–32px halo) to eliminate boundary truncation without exhausting workstation RAM.
4. **Crash-Resilient State Machine:** All file creation follows atomic staging (`data/staging/` → atomic rename to `data/cogs/`). SQLite WAL mode prevents database locking, and an automated startup sweep reconciles interrupted tasks.

---

## 5. Index Build

### Index-Build Procedure
The semantic vector index is constructed incrementally during raster ingestion:

```
Raw GeoTIFF / .SAFE Scene
          │
          ▼
Universal Format Loader & Capability Audit
          │
          ▼
Native Categorical Masking (SCL / QA_PIXEL)
          │
          ▼
True-Color RGB Synthesis & COG Conversion
          │
          ▼
Uniform 224×224px Grid Tiling (30% overlap)
          │
    ┌─────┴─────────────────────────────────┐
    ▼                                       ▼
Valid Pixel Fraction < 60%           Valid Pixel Fraction ≥ 60%
    │                                       │
[SKIPPED — Never embedded]            NaN Imputed to Neutral Mean
                                            │
                                            ▼
                              RemoteCLIP ViT-B/32 Inference
                                            │
                                            ▼
                                512-D L2-Normalized Vector
                                            │
                                            ▼
                                FAISS HNSW Dynamic Insertion
```

1. **Grid Tiling:** The scene's RGB composite is sliced into 224×224px crops matching the RemoteCLIP vision transformer receptive field.
2. **NaN & Cloud Gate:** Any crop with less than 60% valid ground pixels (due to cloud, shadow, or sensor nodata) is discarded from the index to prevent poisoning the HNSW graph.
3. **Inference & Insertion:** Crops are passed through the RemoteCLIP ViT-B/32 image encoder in batches. Output vectors are L2-normalized and added incrementally into the `IndexHNSWFlat` index, persisting vector position IDs into SQLite alongside MGRS coordinates.

---

## 6. Incremental Ingestion

### Incremental-Ingestion Procedure
IRIS eliminates monolithic re-indexing by implementing an incremental pipeline:

1. **Ingestion Workflow 1 (Baseline Scene):**
   * Scene dropped via the UI or folder import.
   * Universal loader parses Sentinel-2 (.SAFE / JP2), Landsat Collection 2, or GeoTIFF.
   * Bands normalized, reprojected, masked, and converted into Cloud-Optimized GeoTIFF format (`data/cogs/`).
   * Semantic embedding crops generated and dynamically appended to the live FAISS index.
   * Scene footprint and metadata registered in SQLite catalog with spatial R-Tree indexing.

2. **Ingestion Workflow 2 (Subsequent Overlapping Scene & Change Trigger):**
   * Steps from Workflow 1 execute identically for Scene B.
   * An R-Tree spatial intersection lookup queries the catalog for previous observations covering the same footprint **from the same sensor**.
   * The nearest chronological prior observation is paired automatically as Scene A.
   * The **5-Phase Change Detection Engine** is enqueued as an asynchronous background worker task:
     * **Phase 1 (Quality Masking):** Mutual validity mask generated (AND operation) from Scene A and B masks.
     * **Phase 2 (Co-Registration):** Sub-pixel geometric alignment using `cv2.findTransformECC` with correlation coefficient (`\rho`) gating and ORB+RANSAC fallback.
     * **Phase 3 (Radiometric Normalization):** Topographic and solar incidence angle correction (TASC) using CartoDEM, plus Pseudo-Invariant Feature (PIF) linear regression.
     * **Phase 4 (Difference & Clustering):** NIR difference channel decomposed via local block-PCA and clustered via K-Means ($K=2$).
     * **Phase 4b (Direction Classification):** Independent classification component matching (IoU $\ge 0.5$) and signed index deltas ($\Delta\text{NDBI}, \Delta\text{NDVI}, \Delta\text{MNDWI}$) categorize changes into Appearance, Disappearance, Expansion, or Contraction.
     * **Phase 5 (Filtering & Scoring):** Morphological opening/closing, multi-year historical persistence check to filter cyclical agriculture/seasonality, and 4-term confidence scoring.

---

## 7. Model Provenance

### RemoteCLIP Foundation Model
* **Backbone:** Vision Transformer ViT-B/32 (3-channel RGB, 224×224 pixel input, 512-dimensional embedding space).
* **Pre-training:** Pre-trained on diverse remote sensing datasets by Liu et al., aligning natural language descriptions with aerial and satellite imagery across varied geographic biomes and ground resolutions.
* **Weights & Framework:** Executed via `open_clip_torch` and PyTorch. Weights are packaged locally in `backend/models/` for 100% offline, air-gapped inference.
* **Device Portability:** Automatically detects CUDA hardware for high-throughput acceleration; gracefully falls back to CPU SIMD vector execution (AVX2/AVX-512) on standard field laptops.
* **Invariance:** Embeddings are L2-normalized, ensuring cosine similarity reduces to simple inner products in FAISS.

---

## 8. Dataset Provenance

IRIS is validated on open-access, publicly accessible earth-observation sources:

| Dataset | Sensor / Platform | Native Resolution | Bands Used | Licence / Attribution |
|---|---|---|---|---|
| **Sentinel-2 L2A** | MSI (Multi-Spectral Instrument) | 10m / 20m | B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR), SCL (Scene Classification) | Copernicus Open Access Policy (Free & Open) |
| **Landsat Collection 2** | OLI/TIRS (Landsat 8 & 9) | 15m / 30m | B02, B03, B04, B05, B06, QA_PIXEL | USGS / NASA Public Domain |
| **CartoDEM** | Cartosat-1 (ISRO) | 30m posting | Surface Elevation & Derived Slope Angle | ISRO / NRSC Bhuvan Open Products |
| **OSCD Benchmark** | Sentinel-2 Multi-Temporal | 10m / 20m | Multi-temporal registered pairs across 24 global cities | Onera Satellite Change Detection Benchmark |

---

## 9. Evaluation Report

Measurements recorded live by the built-in evaluation instrumentation framework (`backend/data/eval_manifest.json`):

### 9.1 Evaluation Hardware & Environment
* **Workstation CPU:** AMD Ryzen 7 250 w/ Radeon 780M Graphics (8 physical cores, 16 logical threads)
* **System Memory:** 15.3 GB Total RAM
* **Execution Mode:** 100% Offline Local CPU Inference
* **Host OS:** Microsoft Windows 11 Enterprise (64-bit)

### 9.2 Measured Benchmarks
* **Indexed Ground Footprint:** 100 km × 100 km standard Sentinel-2 MGRS Tile (`T43RGM`)
* **Indexed Semantic Vectors:** 4,802 crops in live FAISS HNSW graph
* **Full Ingestion Latency:** 363.67 seconds (complete end-to-end pipeline on CPU: extraction, masking, COG conversion, tiling, embedding, and indexing)
  * *Band Extraction:* 98.2s
  * *COG Conversion:* 56.8s
  * *RemoteCLIP ViT-B/32 Inference:* 115.8s (~20.7 crops/second on CPU)
  * *FAISS Dynamic Vector Insertion:* 0.28s
* **Semantic Query Latency:** **< 85 ms** (text query encoding + FAISS HNSW top-50 vector search + SQLite spatial metadata join)
* **Storage Footprint:**
  * *Cloud-Optimized GeoTIFF:* ~280 MB per 4-band 10m scene (DEFLATE compressed)
  * *Vector Index:* ~9.8 MB per 4,800 vectors
  * *SQLite Spatial Catalog:* ~1.4 MB

---

## 10. Setup Guide

### Option A: 1-Click Desktop Launch (Recommended)
IRIS is pre-compiled as a standalone Windows desktop application:
1. Double-click **`IRIS`** on your Windows Desktop (or run [`Launch_IRIS.bat`](Launch_IRIS.bat)).
2. The launcher silently verifies the offline AI backend on `127.0.0.1:8000` and immediately opens the native IRIS desktop application window.
3. No browser, web server commands, or manual terminal management required.

### Option B: Developer Setup (From Source)
Ensure Python 3.10+ and Node.js 18+ are installed.

```bash
# 1. Clone repository
git clone https://github.com/25cseb50robindanie/IRIS.git
cd IRIS

# 2. Setup Python Virtual Environment
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt

# 3. Setup Frontend Dependencies
cd frontend
npm install
npm run build
cd ..

# 4. Launch Desktop Software
Launch_IRIS.bat
```

---

## 11. Demo

* **Desktop Application Window:** Clean, neutral intelligence console with native map rendering, dual-pane swipe comparison, and instant natural-language search.
* **Semantic Retrieval in Action:** Type queries like `"dense industrial facility with storage tanks"` or `"coastal port with maritime berths"` to retrieve matching satellite locations.
* **Explainable Change Detection:** View physical changes categorized by Appearance, Disappearance, Expansion, and Contraction with color-coded confidence indicators and provenance audit cards.

---

*IRIS — Engineered for Operational Geospatial Intelligence.*
