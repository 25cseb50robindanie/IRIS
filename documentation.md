# IRIS — Complete System Documentation

**Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery**
Smart India Hackathon Problem Statement 26227 — Ministry of Defence / Indian Army (DGIS)

*This document is written for someone who has not seen this project before. It explains what the system is, what it does, how every stage works, what we deliberately do not claim, and how it will be measured. Plain English where possible; technical detail where the detail is the point.*

---

## Part 1 — What problem this solves

Satellite archives today are searchable by **metadata**: coordinates, acquisition date, satellite, product type. That means an analyst has to already know *where* and *when* to look before they can look at anything.

Two things are missing:

1. **You cannot search by meaning.** There is no way to ask "show me newly built structures near a river" and get ranked results. You can only ask "show me everything in this box between these dates" and then look through it yourself.
2. **Automatic change detection produces too many false alarms.** Systems that compare two dates and report differences flag clouds, shadows, seasonal vegetation changes, different sun angles, and tiny image misalignments as "change." An analyst who gets a hundred alerts and finds ninety-five of them are clouds stops trusting the system.

IRIS addresses both, with the second being the harder and more important half. The problem statement names ten specific sources of false alarm, and the design's core work is having a specific, explainable answer for each one.

**Hard constraints we work under:**
- Fully offline. No cloud services, no external APIs, no internet during evaluation.
- No login. Single analyst, local desktop application.
- Only public datasets under open licences. The problem statement's permitted list is Sentinel-2, Sentinel-1 (SAR), Landsat Collection 2, and Bhuvan/NRSC products. **What this system does with each differs, and the distinction is deliberate:** Sentinel-2, Landsat and Bhuvan optical products go through the full pipeline including change detection. **Sentinel-1 SAR is scoped to ingestion, cataloguing and display only — it is explicitly excluded from change detection in this version.** SAR requires its own preprocessing chain (decibel conversion, speckle filtering, incidence-angle and layover/shadow correction) that shares almost nothing with the optical path, and none of the optical masking logic — cloud classification, RGB composites, cloud-trust scoring — is even meaningful for radar. Being on the permitted dataset list is not the same as being a supported change-detection input, and this document does not claim otherwise.
- Source code and architecture notes are submission deliverables — the system must be explainable, not a black box.

---

## Part 2 — The technology, and why each piece

| Layer | Choice | Why this one |
|---|---|---|
| Desktop shell | **Electron** | Lets us build the interface with web technology (HTML/CSS/React) while shipping a native desktop app. Debugging uses ordinary Chrome DevTools. |
| Interface | **React + TailwindCSS + shadcn/ui** | Clean, professional components. Visual style deliberately modelled on QGIS — light neutral grey, dense information, no decoration. It is an analyst tool, not a consumer product. |
| Map display | **MapLibre GL** | Open-source, no API key, works fully offline. Smooth pan/zoom over imagery. |
| Backend | **Python + FastAPI** | Python has the geospatial and machine-learning ecosystem this work requires. |
| Reading satellite files | **rasterio / GDAL** | The standard libraries for reading satellite raster formats and their georeferencing. |
| Serving imagery to the map | **titiler** | Explained in Part 4 — this is the bridge between a satellite file on disk and a map you can pan around. Runs entirely on `localhost`. |
| Semantic understanding | **RemoteCLIP** (ViT-B/32) | A CLIP-architecture model additionally trained on remote-sensing image/caption pairs. It is the only realistic option that does **text-to-image** search — see Part 6. |
| Vector search | **FAISS (HNSW)** | Fast similarity search over embeddings, supports adding new vectors without rebuilding the whole index. |
| Catalog | **SQLite + R-Tree module** | One file, no server, works offline. R-Tree makes "which stored scenes overlap this footprint?" fast instead of a full table scan. |
| Background jobs | **Python `multiprocessing.Queue`** | Deliberately *not* Celery/Redis — those require running a second service, unnecessary complexity for a single-user offline desktop tool. |

**Compute:** target hardware is a workstation/laptop with a CUDA-capable GPU. RemoteCLIP runs on GPU where available, CPU otherwise. The CPU path works but is materially slower, and on CPU-only hardware, generating embeddings becomes the ingestion bottleneck. Search itself is fast on CPU either way.

---

## Part 3 — How the pieces fit together, in one picture

```
   Analyst drops in satellite data
              │
              ▼
   ┌──────────────────────────────────┐
   │  INGESTION (runs once per scene) │
   │  detect format → mask clouds     │
   │  → reproject → convert to COG    │
   │  → build RGB → cut tiles         │
   │  → embed with RemoteCLIP         │
   │  → write to catalog              │
   └──────────────────────────────────┘
              │                    │
              │                    └──► visible on map immediately
              ▼
   ┌──────────────────────────────────┐
   │  CHANGE DETECTION                │
   │  (background job, triggered      │
   │   automatically if this scene    │
   │   overlaps an earlier one from   │
   │   the same sensor)               │
   │  align → normalise → difference  │
   │  → cluster → clean → filter      │
   │  → score → store result          │
   └──────────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────────┐
   │  QUERY TIME (milliseconds)       │
   │  Nothing is computed here.       │
   │  Two index lookups only.         │
   └──────────────────────────────────┘
```

**The single most important design decision in this system:** all the expensive work happens **once, at ingestion**. Nothing heavy runs when the analyst searches. This is why queries return in milliseconds — the answers were computed and stored earlier. The trade-off is stated honestly: a change can only be found once its background processing has finished.

---

## Part 4 — Getting imagery onto the screen (and why it needs a helper program)

A satellite file is not a photograph. A `.jpg` already knows how to be drawn. A satellite file (JP2, GeoTIFF) is raw numbers — one grid of measurements per spectral band — plus information about where on Earth those numbers sit. A browser-based map cannot read that directly.

So a small helper program, **titiler**, runs quietly in the background on the same machine. When the map needs to show part of an image, it asks titiler "give me this square as an ordinary picture," and titiler reads the raw satellite file and hands back a normal image tile. The map stitches these together as you pan and zoom.

This is the same technique Google Maps uses — fetching small image tiles as you scroll — except the "server" is a process on your own laptop. **Nothing ever leaves the machine and no internet connection is involved.** There is also no background street map: only the analyst's own imagery, on a blank background.

**Why we convert everything to COG.** A Cloud-Optimized GeoTIFF is the same image saved in a smarter internal layout, so a program can jump straight to the small piece it needs instead of opening a multi-gigabyte file to see one corner. titiler requires this, and the problem statement asks for it. One decision satisfying two requirements.

---

## Part 5 — Walkthrough: importing the first dataset

The analyst clicks **Browse**, picks a folder, and selects a file. Here is everything that happens, in order:

**1. Universal loader.** Accepts `.SAFE`/JP2 (Sentinel-2's native format), GeoTIFF, or COG. Extracts the individual spectral bands.

**2. Capability detection.** An explicit audit step: what bands are present? Is there a quality-assessment band? What is the native coordinate system and resolution? This decides the processing path, and makes "works across different satellites" an auditable decision point rather than logic buried in code.

**3. Quality masking — at native resolution, before anything is resampled.**
- *Sentinel-2 / Landsat:* read the SCL / QA_PIXEL band that ships with the product. Cloud, cloud shadow and snow pixels are marked invalid. This is free, pre-validated information — no model needed.
- *Bhuvan (no quality band):* fall back to a spectral heuristic estimating cloud-like pixels from brightness and index values. **This is an estimate, not reliable cloud detection** — bright concrete, sand and salt flats can look the same. It contributes to a confidence score rather than hard-discarding pixels.

*Why "before resampling" matters:* resampling a continuous measurement (brightness) by averaging neighbours is fine. Doing the same to a category label (cloud / not-cloud) produces meaningless blended values at boundaries. Masks must be derived from data at its original resolution.

**4. Reprojection.** Everything for a given area of interest is put into one common coordinate system. Categorical masks use nearest-neighbour resampling; continuous bands use bilinear/cubic.

**5. COG conversion.** Plus a checksum of the original file recorded for provenance. Deleting the original is an explicit, logged, analyst-visible action — never a silent background cleanup, because the submission requires reproducibility.

**6. RGB composite.** Built from the red, green and blue bands (B04/B03/B02 for Sentinel-2). We always build this ourselves rather than using Sentinel-2's convenient pre-made "TCI" file, because other satellites don't provide one — one code path works everywhere.

**7. Tiling.** Note that **three different tiling concepts exist in this system and are deliberately kept separate:**

| Concept | Size | Purpose |
|---|---|---|
| Embedding crops | Fixed 224×224 px | Input to RemoteCLIP for search |
| Processing windows | Variable strips with a 16–32 px halo | Change detection |
| Display tiles | Set by zoom level | Rendering on the map |

They solve different problems and have no reason to share a size. The halo on processing windows exists because change detection uses connected-component analysis, which can truncate a real feature at a tile seam and lose it entirely. Embedding crops don't have that failure mode — a feature split across two search crops weakens its signal rather than disappearing.

**8. Embedding.** Each 224×224 RGB crop goes through RemoteCLIP, producing a vector, which is added to the FAISS index.

*A guard that matters here:* masked pixels are NaN ("not a number"). Most libraries in this pipeline raise a loud error on NaN — but **a vision transformer propagates NaN silently**. A NaN embedding vector inserted into the search index corrupts distance calculations and returns garbage rankings with no error anywhere. So: crops below a minimum valid-pixel fraction are skipped entirely (an unusable crop should be *absent* from the index, not represented by a made-up vector), remaining NaN is filled before inference, and every vector is checked finite before insertion.

**9. Catalog write.** One SQLite row per tile: tile ID, scene ID, sensor, acquisition date, geographic bounds, a **10-digit MGRS (Military Grid Reference System) coordinate** — the geocoordinate standard used by NATO and Indian Army systems — computed via the `mgrs` Python package (MIT licence, one function call per coordinate), file path, cloud percentage, and a pointer to its vector in FAISS. Bounds are indexed with R-Tree.

**10. Available immediately.** The scene appears on the map as soon as its COG exists — it does not wait for embedding or change detection. Those run in the background. The analyst is never staring at a blank window.

At this point: **one dataset is loaded, viewable, and searchable by text. No change detection has run, because there is nothing to compare against yet.**

---

## Part 6 — How RemoteCLIP actually stores meaning

This is the part people usually guess wrong, and understanding it explains a lot of the rest of the system.

A reasonable guess is that the model tags each image — "this one contains a river." **That is not what happens.** There is no readable label stored anywhere.

RemoteCLIP converts each image into a list of a few hundred numbers — a **vector**. During training it saw millions of satellite image/caption pairs and adjusted itself so that an image's vector and its matching caption's vector land **close together numerically**, while mismatched pairs land far apart. So "river" isn't stored as a word; it is a *region* in a high-dimensional space, and river-like images naturally fall near it because of the geometry the model learned.

The trick that makes text search work: **the same model also converts your typed query into a vector in that same space.** So when the analyst types "newly built structures near a river," the system converts that sentence into a vector and finds which stored image-vectors are mathematically closest. That is the entire search mechanism — convert everything to points in one space, find the nearest ones. No keywords, no tags.

**Why RemoteCLIP and not the alternatives.** TerraMind and Prithvi-EO are both larger, more modern geospatial foundation models. Neither has a text encoder — they were never trained on image-caption pairs, so neither can accept a typed sentence at all. Making them do text search would mean training a text-alignment head from scratch. For this requirement, RemoteCLIP isn't the better option, it's the only working one.

**A real limitation, stated rather than hidden.** A 224×224 crop at Sentinel-2's 10 m resolution covers about 2.24 km × 2.24 km of ground. If that crop contains a village, a river, a forest and one small structure of interest, the embedding describes it by its dominant content — the small feature gets diluted. **GAE Attribution Heatmaps narrow this gap without changing the tile size.** For each top-K result, the system runs one backward pass through RemoteCLIP's ViT layers using Chefer et al.'s Generic Attention-model Explainability method (ICCV 2021, MIT licence), computing which patches the model attended to when matching the query. The output is a 7×7 relevance grid — at 10m imagery, each cell covers ~320m × 320m, so the highlighted area is roughly **0.1 km² instead of 5 km²**, about a 50× reduction. The analyst sees a colour overlay ("attribution glow") on the tile showing *where* within the 2.24km box the match is strongest, rather than searching the whole box manually. This runs at query time only, bounded by the top-K count, and does not affect ingestion speed.

Worth being precise about what this is and isn't: it's patch-level localisation at ~320m, not pixel-level. Don't call it "sub-hectare" (a hectare is 100m × 100m, and the patches are larger than that). It's a genuinely useful precision improvement that avoids the interpolation artifacts that killed the sub-tiling approach, by working *within* the existing tile rather than requiring smaller ones.

**However, the system does more than return the tile and leave the analyst to search within it.** After retrieval, a visual-attribution technique (Generic Attention Explainability, Chefer et al. ICCV 2021) reads back which patches of the ViT the model attended to when matching the query, and renders a colour heatmap overlay on the tile — an "attribution glow" — showing *where within the crop* the match was triggered. For RemoteCLIP's ViT-B/32, this operates at a 7×7 patch grid, where each cell covers roughly 320m × 320m, reducing the analyst's search area from ~5 km² to ~0.1 km² per highlighted patch — a roughly 50× reduction. This runs at query time (one backward pass per result, tens of milliseconds on GPU), uses the same model with no extra weights, and is fully offline. It is not sub-metre localisation — each patch is ~320m across — and the documentation states that rather than overclaiming.

**Therefore semantic search is documented as finding broad categories** ("airfield", "river bridge", "urban edge"), while small tactical-scale features are the job of the change-detection engine. The two capabilities are complementary by design, not redundant.

---

## Part 7 — Walkthrough: importing a second dataset of the same area

Steps 1–10 run identically and independently for the new scene. Then something extra happens.

At catalog-write time, the system checks (using the R-Tree index) whether this scene's footprint overlaps anything already stored. If it does, a **change-detection background job is triggered automatically.** The analyst does not ask for this — importing 2025 data for an area that already has 2022 data *is* the request.

**Two rules govern which scenes get compared:**

1. **Same sensor only.** Footprint overlap alone is not sufficient grounds to compare two images. Sentinel-2, Landsat and LISS-III differ in their spectral response and viewing geometry. NASA maintains an entire separate product (HLS) precisely because making those sensors interchangeable requires BRDF normalisation and spectral bandpass adjustment — neither of which this pipeline performs. Comparing across sensors without that harmonisation would present *sensor difference as change*, producing exactly the confident false positives the whole system exists to prevent. Cross-sensor data remains fully available for viewing and searching; only *differencing* is restricted.
2. **Nearest prior observation, not every prior observation.** The new scene is compared against the most recent earlier scene covering the same ground — not all of them. Comparing everything would grow combinatorially (100 scenes for one area would mean up to 4,950 possible pairs). And it's redundant: if nothing changed between January and April, and nothing between April and July, you already know January-to-July is unchanged. Optionally, if a same-season prior-year scene exists, a second comparison runs against that too, which catches things a purely chronological comparison misses due to seasonal differences.

---

## Part 8 — The change-detection pipeline, stage by stage

This is the core of the system. Everything here is **classical signal processing and statistics — no trained model, no training data.** That is deliberate: for a defence evaluation, an analyst asking "why did you flag this?" needs an answer that points to a formula, not to a neural network's opinion.

### Phase 1 — Quality masking and trust scoring

- **Cloud trust, graded not binary.** Cloud cover is measured strictly *within the area of interest*, not across the whole scene. Rather than rejecting a scene outright, it gets a trust score (`1 − cloud_fraction`), so a 40%-cloudy scene scores 0.60 and stays usable with reduced confidence. A hard reject risks leaving the analyst with *nothing* for a persistently cloudy area — and in monsoon India that is a real scenario, not a hypothetical.
- **Masked pixels become NaN**, establishing a "cleaned baseline" that feeds both search and change detection.
- **The NaN contract.** Every numerical stage downstream rejects NaN — OpenCV's alignment raises "NaN encountered", scikit-learn's PCA and K-Means both raise on non-finite input, and NumPy masked arrays aren't understood by scikit-learn either. So the required pattern everywhere is: carry a validity mask as a first-class array → extract only valid pixels into a flat list → run the algorithm on that → scatter results back into a full-size raster initialised to an explicit "no data" value. **Critically, NaN is never filled with zero in a difference image**, because zero there means "no change" — filling would silently relabel unobserved ground as verified unchanged.

### Phase 2 — Geometric alignment (order matters here)

**1. Grid check — a hard gate.** Verify that coordinate system, pixel origin, affine transform and resolution are *all* identical between the two dates. Same coordinate system is **not** sufficient: two scenes in the same UTM zone can have pixel grids offset by a fraction of a pixel, and differencing them then compares slightly different patches of ground, producing false change along every sharp boundary in the image. If they differ, the moving image is warped onto the reference grid and the check re-runs; if it still fails, the pair is rejected rather than differenced on a best-effort basis.

*(Related: adjacent Sentinel-2 tiles can arrive in different UTM zones, so one coordinate system is chosen per area of interest at ingestion and every acquisition is pinned to it — otherwise a time series over one area would never stack.)*

**2. Mutual validity mask** — the logical AND of both dates' masks, giving pixels that are usable in *both* images. This runs only *after* the grid check passes, because ANDing masks that aren't pixel-aligned combines values from different ground positions and produces a result that is silently wrong rather than obviously broken.

**3. Terrain-risk masking.** Slope is computed from CartoDEM (an ISRO elevation model, available on Bhuvan). Steep slopes get reduced confidence, because when the satellite's viewing angle differs between two passes, mountainsides genuinely look different without anything having changed. The confidence multiplier is `cos(slope_angle)`, floored at roughly 0.50 beyond 60° — cosine rather than an arbitrary curve because both view-angle foreshortening and slope illumination are cosine relationships in the incidence angle, so the discount follows the same geometry as the error it's discounting. Floored rather than decaying to zero so that very steep terrain heavily discounts a detection instead of silently erasing it; those candidates are flagged for manual review instead. If elevation data isn't available for an area, this term defaults to neutral rather than failing the pipeline.

**4. Co-registration.** Primary method is ECC (Enhanced Correlation Coefficient), which iteratively nudges the second image until it best matches the first. Three failure paths are handled:
- Two *loud* failures (NaN encountered; non-convergence) throw exceptions — caught, escalated to fallback.
- One *silent* failure, which is the dangerous one: **ECC can converge on a wrong answer and return a transform with no error at all**, typically over low-texture areas like open water or bare fields where there's nothing distinctive to lock onto. A silently bad alignment injects false change across the entire scene. So the correlation coefficient (`rho`) that ECC returns is **thresholded** — below the threshold is treated exactly as a failure, regardless of no exception being raised.
- Fallback is ORB feature detection (finding distinctive corners) + RANSAC (discarding bad matches). If that also fails its quality bar, the pair is rejected and flagged for manual review rather than differenced on an alignment nothing trusts.

**5. Block-level alignment confidence.** Rather than one score for the whole image, alignment quality is scored per small block. Low-scoring blocks are marked "uncertain zones" and cannot raise a change alert on their own. This matters because misalignment is usually *local* — worse near hills, water, or featureless ground — so a single whole-image score would either over-trust or over-discard.

### Phase 3 — Radiometric normalisation (making the two images comparable)

- **Temporal median compositing (optional).** Where several dates exist for one location, taking the per-pixel median across them suppresses residual shadow and cloud noise that masking missed. Shadows move between dates; real ground features don't, so the median votes out the transients.
- **Topographic correction (TASC).** Using elevation data and the sun angle recorded in the product metadata, compute how illuminated each pixel actually was and correct for it.
- **Pseudo-Invariant Features (PIF).** This is how seasonal and lighting differences get cancelled. The system finds pixels that *shouldn't* have changed — selected statistically by low variation over time **and** high local registration confidence, not from a hardcoded list of materials — measures how much brighter or darker they look between the two dates purely due to sun angle, season or sensor drift, then rescales the whole image to cancel that out. If too few reliable anchors are found, PIF is skipped and the correction relies on topographic correction alone, with that fallback recorded so the confidence score reflects it.

### Phase 4 — Detecting the change

**Which bands are used** is specified, not left open: *not* all 13 Sentinel-2 bands (adjacent bands are highly correlated, and the redundancy reduces sensitivity), and not the 60 m atmospheric bands (noise only). Change magnitude uses the four 10 m bands; the primary detector runs on a single difference image (near-infrared difference by default, for maximum vegetation/built-up contrast).

- **Subtraction.** Difference = new image − reference image. Gated by the grid check.
- **Block-PCA**, following Celik's 2009 method. An implementation detail that's easy to get wrong: this method's "PCA" is a *local-neighbourhood feature extractor on one difference image*, not a spectral decomposition of a multi-band stack. It partitions the difference image into blocks, extracts eigenvectors from those, and projects each pixel's neighbourhood into that space.
- **K-Means clustering with K=2**, splitting every pixel into "changed" or "unchanged" — with the threshold found from the data itself rather than pre-decided, so it adapts to the landscape.

  *A required rule:* K-Means returns two **unlabelled** clusters, and which one means "changed" is arbitrary — it will silently swap between scenes if read from the cluster index. So the change cluster is identified explicitly as the one with the **larger centroid magnitude**, since difference values where change occurred are higher than where nothing changed.

**Known characteristic, designed around rather than hidden:** this method is documented as noise-sensitive and false-alarm-prone. That is exactly why Phases 1–3 and the filters below are load-bearing, not optional polish.

### Phase 4b — Which *direction* did it change?

The problem statement requires identifying **appearance, disappearance, expansion or contraction**. A binary changed/unchanged mask cannot answer that — K-Means clusters *magnitude* and discards the sign. Direction is reconstructed from three signals:

*A naming note that matters, because two different kinds of object exist in this pipeline and confusing them produces different results:*

| Name | Built from | Used for |
|---|---|---|
| **Classification Components** | Each date **on its own**, by index thresholds | Working out change *direction* (this phase) |
| **Change Blobs** | The **binary change mask** | Cleanup, shape classification, confidence scoring (Phase 5) |

The fixed order is: **Classification Components → direction classification → binary change mask → opening → closing → confidence scoring.** Morphological cleanup touches Change Blobs only and **never modifies Classification Components** — direction classification is finished before any morphology runs, so cleanup cannot retroactively alter what was labelled an appearance or a disappearance.

1. **Classification Component matching.** Label connected components in each date separately (classified by index thresholds — all named calibration parameters, with per-scene Otsu thresholding preferred over fixed values), then match them between dates by overlap. Present in the new date only → **appearance**. Present in the old date only → **disappearance**. Matched but grown or shrunk → **expansion/contraction**, requiring the area change to exceed alignment error plus a tolerance so jitter never reads as growth.
2. **Signed index deltas.** The mean change in NDVI/NDBI/NDWI inside each blob names the change: built-up index up and vegetation index down → construction; vegetation up → regrowth; water index up → flooding. This is the most explainable signal and needs no training.
3. **Change Vector Analysis** as a cross-check — with the caveat that different real changes can share the same vector angle, so direction labels always carry a confidence score rather than being asserted.

*Band note:* NDBI and MNDWI need the 20 m SWIR band alongside 10 m bands. SWIR is resampled up to 10 m with **bilinear** interpolation onto the identical shared grid for both dates — never nearest-neighbour, because a nearest-neighbour block edge can shift a whole 20 m cell between acquisitions and manufacture false change along every edge. And the effective resolution of any SWIR-derived index stays 20 m; we state that rather than implying 10 m precision.

### Phase 5 — Cleanup, filtering, scoring

- **Morphological cleanup — four operations, not two, and on Change Blobs only.** Opening (erode then dilate) removes speckle noise. Closing (dilate then erode) fills small internal holes. These are *different* compound operations and doing only one delivers only one of the two effects. Structuring element is small (3×3) at 10 m; larger erases genuine small changes. Two side effects respected: opening can sever thin linear features, so the linear-feature classification path runs on the pre-opening Change Blobs; closing can merge nearby distinct Change Blobs, which distorts the blob counts and areas used for shape classification and confidence scoring here. Neither can affect direction classification, which already completed in the previous phase on a separate set of objects.
- **Seasonal persistence filter.** For every surviving Change Blob whose mean vegetation-index drop exceeds a named calibration parameter (`NDVI_DROP_SIG`, starting value 0.15), query the catalog for the same area, same calendar month, across available prior years. A *cyclical* drop (down every winter, back every spring) is discarded as seasonal. A *step change* that breaks historical pattern is retained as real clearance. This distinction matters: without it, a genuine deforestation event and ordinary winter dieback look identical, and the filter would suppress real detections.
- **Confidence score**, computed per candidate polygon after cleanup:

  ```
  Confidence = 0.35 × NormRMSE + 0.35 × NormClusterDist
             + 0.15 × TerrainFlatness + 0.15 × ValidCoverage
  ```

  All four terms are normalised to [0,1]. Alignment error uses a saturating map. Cluster distance uses its **per-scene percentile rank** — necessary because that quantity is unbounded and scene-dependent, so any fixed normalisation would make scores non-comparable between scenes and silently break the review queue's ranking (candidates from different scenes sit in one queue). **The weights are illustrative starting values to be calibrated, not derived constants.**
- **Insufficient-evidence floor.** If a candidate's mutual valid coverage falls below a minimum, it is dropped with an explicit "Insufficient Evidence" flag. This is a *hard gate* in addition to the weighted coverage term — a detection shouldn't reach moderate confidence just because the other three terms looked fine on the small sliver of tile that was actually usable.
- **Earliest observation.** Walk back through the catalog to find the earliest date the change is visible, skipping dates whose usable coverage is too low — the oldest date on file is not the answer if it's mostly cloud.
- **Classification.** Vegetation clearance / water-extent change / construction, from spectral indices. Snow flagged separately.

**Everything above is stored**, including low-confidence candidates, tagged rather than discarded. Filtering by confidence happens at review time, so a mis-calibrated threshold can be revisited without reprocessing.

---

## Part 9 — What the analyst actually sees

1. **Launch** — opens directly to a map canvas. No login.
2. **Browse** — a native folder picker; select imagery. The right-side panel shows a live vertical step list as the pipeline runs: Capability Detection → Masking → Reprojection → COG Conversion → Tiling → Embedding → Indexing, each with an icon state and live progress counts. The analyst sees real processing stages, not a spinner — and a judge watching the demo sees the pipeline architecture working in real time.
3. **Map** — pan and zoom over the imagery. **Always the human-readable true-colour composite** — infrared and index math happens invisibly in the pipeline; the analyst is never shown a raw infrared band by default. By default the map shows the most recent available coverage for wherever you're looking, stitched continuously — never multiple dates stacked on top of each other.
4. **Search** — type a query, get ranked results with confidence. Filters for area, date range and sensor. If nothing clears the similarity threshold, the system says **"no confident match"** rather than returning the closest thing it has.
5. **Click a result** — the map flies to that location and highlights it. The retrieval tile shows a **colour attribution overlay** indicating which part of the 2.24km² crop triggered the match — the analyst doesn't have to search the whole tile visually.
6. **Before/after comparison** — two pannable map panels, each with its own coordinate/scale/CRS readout, detected change regions outlined as overlays. Note that the *highlight* is what communicates the change — the analyst isn't asked to spot it by squinting at pixels.
7. **Change detail** — confidence score, change type, direction, earliest-observation date, and which factors drove the score.
8. **Confirm / Reject** — writes to the audit log with timestamp and operator.
9. **Find Similar Sites** — from any result, discover comparable locations elsewhere in the archive.

**When search and change detection appear to disagree**, there's an explicit policy, because this *will* happen: RemoteCLIP might strongly match "newly built structures near a river" while the change record says no significant change. These aren't in conflict — RemoteCLIP has no concept of time and cannot verify the word "newly" from a single snapshot. So the two are never merged into one score; temporal query words boost results with confirmed change records but never hide the others; and "analysis pending" is kept distinct from "no change," because those mean very different things.

---

## Part 10 — Discovery, provenance and export

**Discovery/clustering.** Two mechanisms: per-query nearest-neighbour ("find similar"), plus periodic archive-wide clustering so groupings exist before anyone asks. Clustering uses K-Means with silhouette-selected *k* on normalised embeddings. We chose this over the more fashionable UMAP→HDBSCAN pipeline deliberately: a stochastic nonlinear projection feeding a density clusterer is hard to defend when a reviewer asks *why* two sites were grouped, and explainability is a hard constraint here. K-Means centroids answer that question directly.

**Visual Evidence Card.** Clicking any confirmed detection generates a single-page printable HTML summary — what a military intelligence analyst would call a "product": before/after thumbnail chips, MGRS coordinate, change type and direction, confidence score with the four-term decomposition visible, sensor metadata, acquisition dates, SHA-256 raster checksums (already in the catalog from ingestion), and the analyst's sign-off with timestamp. This is a formatted view of data the system already stores, not a separate generation pipeline — one HTML template populated from existing provenance fields. It exists because the output of this system is an intelligence product, not a search result, and the distinction matters for how it's used downstream.

**Visual Evidence Card.** Clicking any confirmed detection generates a single-page printable HTML summary: before/after thumbnail chips, MGRS coordinate, change type and direction, confidence score with four-term decomposition, sensor metadata, acquisition dates, SHA-256 raster checksums, and analyst sign-off. This is a formatted view of data the system already stores — one HTML template, not a new computation. Its value is operational: it produces a standardised intelligence summary suitable for passing up a command chain, not a data dump requiring GIS software.

**Audit trail.** Every confirm/reject decision stored with timestamp, operator identifier and optional notes.

**Export format: GeoJSON** with explicit CRS, one feature per candidate, each carrying **MGRS grid reference**, source scene IDs for both dates, acquisition timestamps, sensor, the processing record (which masking path, which PIF fallback tier, alignment method and error, retained components), the confidence score with its four component terms, the direction classification with its confidence, and the analyst's decision.

**What is in MVP scope here, and what is deferred.** Stated plainly so an evaluator knows which parts of this section describe built behaviour versus planned behaviour:
- **In MVP:** confirm/reject actions, SQLite persistence of every decision, and GeoJSON export with the full provenance block.
- **In MVP, and instrumented from the first commit:** automated evaluation-report generation (build time, storage footprint, indexed area, tile counts, query latency, hardware). This cannot be bolted on at the end — build time and latency cannot be reconstructed retroactively, so the logging goes in as each stage is written.
- **Deferred beyond MVP:** feedback-driven reranking, where confirm/reject history improves future result ordering. The decision history is captured from day one so this can be added later without re-collecting data, but the reranking logic itself is not built in this version.

**Reliability engineering.** Because a desktop app gets closed mid-job: every job carries an explicit state (`queued → processing → completed/failed`), a unique constraint on the scene pair so nothing is processed twice, outputs written to a staging directory and atomically renamed into place only on success, and a startup sweep that resets interrupted jobs and deletes orphaned staged files. SQLite runs in WAL mode with a busy timeout, connections are opened inside each worker process after forking (never inherited — that can corrupt the database silently), and write transactions use `BEGIN IMMEDIATE`.

---

## Part 10b — What makes this different from the other 499 teams

Every individual technique in this system — SCL masking, PIF normalisation, ECC alignment, NDVI — is established, published, searchable. If 500 teams research this problem, many will find the same pieces. What differentiates is not any single technique:

1. **Ablation Transparency.** Every other team will *claim* their system reduces false alarms. This system can *show it* — toggle the suppression stack off in the UI and watch false positives flood the map, toggle it back on and watch them disappear. This is the single most convincing demonstration possible, because it either works or it doesn't, live, in front of the panel.

2. **Negative-Evidence Confirmation.** "We confirmed nothing was built here, on these dates, at this confidence" is an intelligence product. "No results" is not. Most systems only report what they found; this one reports what it looked for and didn't find, with the same provenance as a positive detection.

3. **Decision Trace.** Every detection carries its full reproducible processing chain — which masking path ran, which alignment method and its error, which PIF fallback tier, four confidence terms, direction classification. Not a feature — an accountability mechanism. "Every detection IRIS produces is independently reproducible" is a sentence worth saying in the demo.

4. **GAE Attribution Heatmaps.** Instead of returning a 2.24km box and leaving the analyst to search it, the system highlights *where within the box* the match is strongest, using the model's own attention — a ~50× reduction in search area that no team treating their embedding model as a black box can replicate.

5. **Temporal Coverage Heatmap.** Shown *first*, before any query: "here's how well this region has been observed." Every subsequent result is more credible because the judge already knows where the system is confident and where it's thin.

These are not features bolted on for presentation — each one is a direct consequence of architectural decisions (precomputed analysis, continuous confidence, existing provenance fields, ViT backbone access) that most teams won't have made.

---

## Part 11 — What we deliberately do NOT claim

This section exists because a system that overstates itself fails the first hard question. Every item here is a considered scope decision with a reason.

| We don't claim | Why |
|---|---|
| **Individual vehicle detection** | At 10 m, a vehicle is smaller than one pixel. The information is physically not in the data. "Vehicle concentration" queries are interpreted as anomalous ground-texture patterns, not object counting. |
| **Verified road mapping** | Published work reaches usable road maps from Sentinel-2 only by super-resolving to 2.5 m; plain 10 m gave roughly 37% IoU even with a trained deep network. We surface elongated blobs as low-confidence "possible linear feature" with shape descriptors, not confirmed roads. |
| **Super-resolution enhancement of evidence** | Super-resolution invents plausible detail rather than recovering real information. For a defence evaluation, an "enhanced" image showing texture that isn't there is a liability. Display-only if used at all; never feeds a decision. |
| **Cloud/haze removal** | Same principle. We never reconstruct what's hidden — we find a genuinely clearer observation from another date instead. |
| **Cross-sensor change detection** | Requires BRDF and bandpass harmonisation we don't perform. Restricted rather than done badly. |
| **Sentinel-1 SAR** | Needs its own preprocessing pipeline (speckle filtering, incidence-angle/layover correction). Architected as an isolated deferred path, analyst-override only. |
| **Full terrain-shadow correction** | Mitigated via elevation-based risk masking and topographic correction, not eliminated. Steep terrain carries reduced confidence by design. |
| **Live analysis of brand-new imagery** | Change detection runs at ingestion. A change is findable once processing completes, not the instant it's queried. |

**Resolution ceiling, stated plainly:** 10 m is the best available across *every* permitted dataset source. The organisers set that list, so this constraint applies identically to every team and the evaluation itself must be built on the same data.

---

## Part 12 — How this gets measured

Every numeric threshold in this design — cloud trust curve, the ECC `rho` gate, the 0.5 px alignment trigger, PCA component retention, the confidence weights, the coverage floor, morphological kernel size, minimum blob size, and all index thresholds — is a **domain-standard starting point, not a validated value.**

**Two-stage validation:**
1. **OSCD** (Onera Satellite Change Detection) — 24 Sentinel-2 image pairs, pixel-level ground truth for 14 of them, free via IEEE DataPort. Run the pipeline, measure real precision/recall/F1, set thresholds from results.
2. **Local re-verification** — re-check on a real Indian landscape (Noida International Airport, Jewar: farmland 2018–19 → major construction 2023–24) to confirm OSCD-tuned thresholds behave sensibly on local terrain and land-use patterns.

**Realistic expectations, so nobody is surprised.** On OSCD-class data, strong *supervised* deep-learning methods reach roughly F1 0.46–0.53. *Unsupervised* methods reach roughly F1 0.33–0.46. A classical unsupervised pipeline like this one should be expected around **F1 0.2–0.4**. Anyone promising high accuracy for unsupervised change detection at this resolution is either using different metrics or hasn't measured yet.

**Ablation is required, and it is the measurement that actually matters.** Reporting precision/recall for the whole pipeline shows the system works. It does **not** show that the false-alarm suppression machinery contributes anything — which is this project's central claim. So the same imagery is run with stages switched off, one at a time: the full pipeline; suppression stack entirely off (raw difference only); quality masking off; radiometric normalisation off; co-registration off. Each reported as precision/recall/F1 plus raw false-positive count.

This is the direct answer to the hardest fair question anyone can ask — *"how do you know your false-alarm suppression actually suppresses false alarms?"* — and it cannot be answered by design reasoning, only by measurement. It also has teeth: if disabling a stage doesn't measurably degrade results, that stage isn't earning its place, and we say so rather than keeping it for appearance.

Alongside OSCD, a small set of Sentinel-2 pairs over Indian terrain is hand-annotated — labelling real change and, where a detection is wrong, *which* false-positive source caused it (cloud / shadow / seasonal / illumination / misregistration). That per-source labelling is what lets the ablation attribute improvements to specific stages rather than just showing an aggregate number move.

**Also instrumented from the first commit** (because build time and latency cannot be reconstructed retroactively): per-stage build time, storage footprint, indexed area, tile counts, query latency distributions reported *separately* for precomputed lookups versus on-demand pair analysis, and hardware used — all emitted as a machine-readable manifest alongside checksums and version pins.

---

## Part 13 — Honest assessment for an evaluator

**What is genuinely strong here:**
- Every one of the ten false-alarm sources the problem statement names has a specific, named, explainable mechanism — not one generic "AI handles it."
- Confidence degrades continuously rather than making binary decisions, with a hard floor for insufficient evidence.
- Crash-class and silent-failure paths have been deliberately hunted: NaN propagation, K-Means label inversion, ECC silent misconvergence, grid misalignment, database corruption across process forks.
- The decision record documents *rejected* alternatives with reasons, not just what was chosen.
- Claims are scoped to what the data physically supports.

**What is not yet known, and cannot be known from a document:**
- **No code has been written and nothing has been calibrated.** Design coherence and working software are different things.
- The seasonal persistence filter currently has no minimum-observation requirement, so it can run on sparse history — and its failure mode is *false negatives*, which an evaluation surfaces less readily than false positives. This is a known open item.
- Topographic correction runs before PIF anchor selection, which can contaminate anchors on steep slopes. Known open item; accuracy risk, not a crash.
- Classification Component matching by overlap is unstable for features only a few pixels across — a 1–2 pixel shift on a 3×3-pixel object destroys the overlap score. Known open item.
- PIF anchor selection by low temporal variance is defensible in spirit but is not the published standard (IR-MAD is). A considered choice, not an oversight.

**On novelty, honestly:** this is careful systems engineering that integrates existing tools well. It is not a new algorithm, and shouldn't be presented as one. The problem statement asks for a working system evaluated against held-out queries and labelled change cases — not a novel method. What differentiates it is completeness of coverage, a documented decision trail, and measured rather than claimed performance.

---

*Companion document: `architecture.md` (v2.2.7) holds the full technical specification, citations and decision log.*

</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-09-20T14:09:47+05:30.
</ADDITIONAL_METADATA>