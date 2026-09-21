# IRIS — handoff notes (what is NOT in the committed code)

Written 2026-09-21 by the AI assistant that built Milestones 2–4 with the user. Read `AGENTS.md` (rules) and
`architecture.md` (design) first; this file covers everything they do not: state of the repo, decisions, bugs found and
fixed, workarounds, environment traps, and how to verify each feature.

---

## 0. State of the repo (read this first)

- Git: `main`, last commit `265cf74 Milestone 3`. **Everything below is uncommitted**: the Milestone 4 "demo readiness"
  work (merge, MGRS, export, Find Similar, instrumentation, decision trace, wider comparison view) plus the two
  follow-up fixes (Find Similar error visibility; large-area merge with size guard; area sort).
  `git status` lists ~27 modified and ~14 new files. Commit before doing anything else, ideally as two commits.
- `backend/data/eval_manifest.json` shows as untracked: it is generated at runtime. Add `backend/data/` (or at least
  that file) to `.gitignore`.
- Backend test suite: **185 passing** (last full run 2026-09-21, ~6.5 min). Frontend: `npx vite build` succeeds.
- The user's real catalog lives in `backend/data/` (2 scenes, Sentinel-2 tile T43RGM, 2022-06-23 and 2026-08-21, the
  Jewar airport area; 4,802 tiles). **Its change candidates predate merging** (5,000 unmerged blobs). They only become
  merged detections after the pair is re-run (see §7, "Re-running a pair").
- The user's own backend process must be **restarted** to pick up new routes (`/api/search/similar`, `/api/export/...`,
  `/api/eval/manifest`). A backend started earlier answers 404 for them (this is very likely why "Find Similar did
  nothing" was reported).

---

## 1. Run, build, test

Always use the venv interpreter explicitly: bare `python` on this machine resolves to the Microsoft Store stub.

```bash
# backend (from backend/ — paths are cwd-relative, see §4)
cd backend && ../.venv/Scripts/python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
# frontend dev server (browser dev mode: no Electron, file picker is a window.prompt())
cd frontend && npm run dev            # http://localhost:5173 (127.0.0.1:5173 also allowed by CORS)
cd frontend && npx vite build         # production build check
# Electron shell exists at frontend/electron/{main.js,preload.js}: npm run electron:dev

# tests (from backend/)
cd backend && ../.venv/Scripts/python -m pytest -q -p no:cacheprovider            # everything, ~6.5 min
../.venv/Scripts/python -m pytest -q tests/test_grouping.py                        # ~40 s, pure + 3 pipeline runs
../.venv/Scripts/python -m pytest -q tests/test_direction.py tests/test_demo_features.py  # embedding-heavy, minutes
```

`tests/conftest.py` has an autouse fixture that `chdir`s every test into a temp dir, so tests never touch
`backend/data`. Tests that ingest through `POST /api/ingest` run the RemoteCLIP embedding for real (slow, needs
`checkpoints/RemoteCLIP-ViT-B-32.pt`). Tests that only need the change pipeline use
`tests/s2_factory.py::ingest_without_embedding` (fast).

Black is **not installed** in the venv; formatting has been matched by hand (~120 columns). No linter has been run.

---

## 2. Module map

### Backend (`backend/`)
| Path | Purpose |
|---|---|
| `main.py` | FastAPI app, titiler mount at `/tiles`, CORS (5173 only), no-cache on tilejson, QGIS DLL/PROJ_DATA setup, startup requeues jobs left `processing`. Import order matters (see §4). |
| `instrumentation.py` | Evaluation manifest: per-stage timings, storage, index stats, query latency percentiles, hardware. Writes `data/eval_manifest.json` after every ingest / change job / query / scene delete. |
| `mgrs_ref.py` | `to_mgrs`, `bounds_centre_mgrs`, `backfill_missing`. Precision constant `MGRS_PRECISION = 5` (10 digits). |
| `api/ingest.py` | `POST /api/ingest`, embedding background job, per-tile MGRS. `EMBED_LOCK` serialises embedding + change detection. |
| `api/pipeline_status.py` | In-memory import progress (`pipeline_tracker`), now also records per-step start/end for instrumentation. |
| `api/search.py` | `POST /api/search` (text → tiles + change results), models shared by other routers. |
| `api/similar.py` | `POST /api/search/similar` (Find Similar). |
| `api/change_search.py` | Change-aware half of search: overlap with tiles, after-crop match percentile, direction boost, negative-evidence meta (`area_bounds`, `mgrs`, `dates_analysed`). |
| `api/change.py` | `GET /api/changes` (filters/sort/cap/area), `/changes/jobs`, `/changes/{id}` (detail incl. `display_bounds`, `processing_details`), review POST. |
| `api/change_export.py` | `GET /api/export/changes` GeoJSON download. |
| `api/provenance.py` | Shared wording/fields for the decision trace and export `processing` block; `display_bounds`, `intersect`, `analysed_area`. |
| `api/evaluation.py` | `GET /api/eval/manifest`. |
| `api/scenes.py` | Scene list / delete (rebuilds FAISS, remaps ids, removes files). |
| `api/catalog_status.py` | `/api/catalog/status` used by the frontend to resume state and gate search. |
| `catalog/database.py` | SQLite schema + idempotent migrations (`_ensure_columns`), scenes, tiles (`mgrs_ref`), footprints R-Tree. |
| `catalog/changes.py` | Jobs, candidates (`finish_job` writes candidates + status atomically), reviews, `query_candidates`. Normalises legacy type/direction and derives MGRS for old rows. |
| `ingestion/loader.py` | Generic raster ingest, COG conversion, **`io_path` / `plain_path`** long-path helpers. |
| `ingestion/safe.py` | Sentinel-2 L2A `.SAFE` reader: bands B02/B03/B04/B08 → 4-band analysis COG, SCL → 10 m, RGB display COG. |
| `embedding/` | `crop.py` (224 px tiles, skip >50% nodata), `embedder.py` (RemoteCLIP ViT-B/32, offline), `index.py` (FAISS HNSW inner product; `vectors_for`, `search_subset`, rebuild-on-delete), `progress.py`. |
| `change_detection/params.py` | **Every tunable**, with comments. Start here to change behaviour. |
| `change_detection/pipeline.py` | Orchestrator; phases: masking → grid_check → alignment → radiometry → detection → grouping → scoring → direction → merging. `details` JSON is the decision trace. |
| `change_detection/masking.py` | SCL validity, CloudTrust. `alignment.py` ECC/ORB (skipped for same MGRS tile). `radiometry.py` PIF robust normalisation. `detection.py` difference image, K-means k=2, morphology, labelling, per-date class rasters. `direction.py` Phase 4b. `scoring.py` confidence + `finalize_candidates`. `grouping.py` merging + polygons. `rasters.py` strip reader helpers. `trigger.py` pair planning. |
| `tests/` | `s2_factory.py` builds synthetic .SAFE products (see §8); one test file per area. |

### Frontend (`frontend/src/`)
`App.jsx` (all state: scenes, search, changes, filters, review, Find Similar, polling) · `api.js` (`API_BASE`) ·
`mgrs.js` (client MGRS, same precision as backend) · `download.js` · `components/`: `Sidebar` (semantic results, hosts
`ChangeResults`), `ChangeResults` (filters, cards, negative-evidence card, export menu), `ChangeComparison` (before/after,
review buttons, Processing Details accordion), `CompareMap`, `MapView`, `ExportMenu`, `SectionHeader` (has `actions`
slot), `StatusBar` (hover coords + MGRS), `WorkspaceStatus`, `PipelineProgress`, `SceneMenu`, `Toolbar`, `SearchBar`,
`changeKinds.js` (labels, icons, type/direction options, `hectares`, `describeDirection`).

### Repo root oddities
`main.py`, `database.py`, `main.js`, `preload.js`, `package.json` at the root are stray/older copies (the root `main.py`
is NOT the app: `import main` from the repo root loads it instead of `backend/main.py`). `data/` at the root is old
Milestone-1 output. `checkpoints/RemoteCLIP-ViT-B-32.pt` is required and is found relative to the project root.

---

## 3. Decisions not in `architecture.md`

**Confidence (scoring.py).** `0.35·alignment + 0.35·cluster-distance percentile + 0.15·terrain + 0.15·valid coverage`.
- Alignment for same-MGRS-tile pairs is a fixed **0.90** (grid identical by construction, not measured). Terrain is a
  placeholder **1.0** (no DEM). These constants create a ~0.47 floor, so real scores span roughly 0.63–0.97.
- The percentile was first ranked against pixels (saturated at ~1.0), then against blobs, and is now ranked against **one
  peak value per detection** (its strongest blob). Ranking against every fragment let merged weak changes borrow their best
  fragment's rank and squeezed the scale upward.
- A merged detection takes the **best sub-blob's confidence and that blob's four terms** (so the breakdown still sums).
- Storage: threshold `MIN_STORED_CONFIDENCE=0.3` and the 5,000 cap are applied **after merging**. When the cap bites,
  `STORED_AREA_RESERVE=500` slots go to the largest detections regardless of confidence.
- Display cap is 200 per pair; with `sort=area` the cap keeps the 200 **largest**, otherwise the 200 most confident.
- MMU is `MIN_BLOB_PIXELS=100` (1 ha). Morphology is 3×3 open then close.

**Direction (Phase 4b, direction.py).** Per-date classes from NDVI/NDWI thresholds (`params.py`); water wins over
bare/veg. No SWIR band is ingested, so "bare with higher NDBI" is a proxy (brighter visible bands + falling NDVI) and
"construction" is a best guess. `NDVI_DIRECTION_MIN=0.05` (not 0.15) because bare pixels are already <0.15. Direction is
computed per blob, then per merged detection by **area-dominant named type** ("unclassified" never outvotes a named type).
Expansion/contraction rarely fire end-to-end because blobs contain only changed pixels (unit-tested on `decide`).

**Merging (grouping.py) — read the size-guard bullet.**
- Levels, widest first: large-area dilation 30 px then 20 px (group merges only if ≥3 blobs or >500 px), then near
  dilation 10 px then 5 px (any close blobs). A blob takes the widest level whose group qualifies, else stays single.
  Merge distance is ~2× the dilation (blobs' gaps up to 60/40/20/10 px).
- **Size guard at every level** (`LARGE_AREA_MAX_EXTENT_PX=500`, `LARGE_AREA_MAX_PIXELS=250000`): without it, merging
  chains transitively across dense scenes (real pair: one 151,937 ha detection at 30 px, one 94,522 ha at 10 px, 46 MB
  outline). This deviates from the literal request ("merge groups of 3+ or >500 px") on purpose; tell the user if the
  airport-scale change should exceed 5 km / 2,500 ha and raise the limits.
- Geometry (blob pixels, areas, outlines) always comes from the undilated mask. Outlines are traced with
  `rasterio.features.shapes` on the label raster, reprojected to WGS84 (6 dp), rings wound per RFC 7946, stored as JSON in
  `change_candidates.geometry`. Centroid is the pixel-mean of the changed pixels; MGRS is derived from the *rounded* stored
  centroid so they always agree.

**Search.** Semantic top_k=10. A change "matches" when the after-date crop under it is in the top 25% (`0.75` percentile)
of that scene's tiles for the query. Combined score `0.5·match + 0.5·confidence (+0.2 direction boost)`. Direction
keywords are whole-word (`direction_hints`). Search scope = the selected scene plus overlapping same-sensor scenes.

**Find Similar.** Seed vector comes from FAISS (`vectors_for`), searches the whole archive, seed excluded. A change
candidate seeds with the scene-B tile containing its centroid (else max overlap). Tile-id pattern is ASCII on purpose.

**MGRS precision = 5 (10 digits, 1 m).** The brief said both "10-digit" and `MGRSPrecision=4` (8 digits). One constant
in `mgrs_ref.py` and one in `frontend/src/mgrs.js`.

**Comparison window.** `DISPLAY_WINDOW_FACTOR = 4.0` × the blob box in total (1.5 boxes of context per side). The brief
said "4×" and "2× padding each side" (=5×); 4× was chosen. Clamped to the intersection of both scenes' extents; the window
slides inward at edges and shrinks only if the scene is smaller.

**Export.** Route is `/api/export/changes`, not `/api/changes/export`: FastAPI would match `/changes/{candidate_id:int}`
first and return 422. Legacy candidates without outlines export their bounding box with `geometry_source="bounding_box"`.
Export ignores UI filters (it is the audit trail).

**Reviews.** Analyst id = OS username (`getpass.getuser()`), single-analyst tool, no auth. The UI updates optimistically
and rolls back if the POST fails.

**Instrumentation.** Query latency = time inside the request handler (includes text encoding), not network. Ingest stage
times come from `pipeline_tracker` step start/end; change stages come from the job's `timings_s`, renamed
(`radiometry`→`normalisation`, `direction`→`classification`). Manifest is hydrated from the file on restart.

**Scene delete** rebuilds the FAISS index (HNSW cannot delete), remaps `faiss_id`s, refuses while an import holds
`EMBED_LOCK`, and removes files. Old vectors of a re-imported scene remain in the graph but are filtered by the catalog
join (searches over-fetch ×5 to compensate).

**Legacy data.** Candidates with old type names (`vegetation_loss/gain`) or NULL direction are *presented and filtered* as
"unclassified"; rows without `mgrs_ref` get one from the bbox centre; rows without geometry export a box.

---

## 4. Environment traps and workarounds currently in place

1. **cwd-relative data paths.** `data/…` is resolved against the process cwd (loader, database, index, instrumentation).
   Run uvicorn and pytest from `backend/`. Starting from the repo root uses `data/` at the root (old, empty-ish).
2. **`import main` from repo root loads the stray root `main.py`.** Run from `backend/`.
3. **Windows MAX_PATH (LongPathsEnabled=0).** Python `open/stat/glob` fail beyond 260 chars (nested `.SAFE` folders, deep temp
   dirs) and surface as a misleading "File not found". Use `ingestion.loader.io_path()` for Python file I/O and
   `plain_path()` for GDAL/display. Deleting deep scratch directories from PowerShell also fails: create a new directory
   instead of cleaning the old one.
4. **QGIS-based venv.** `.venv` was created with `--system-site-packages` from `C:\Program Files\QGIS 3.44.14\apps\Python312`
   (originally at a path containing an apostrophe, `C:\Danie's\Projects\IRIS\.venv`, hence the broken junction the user
   once had; `backend/data` is a normal directory now). Some packages come from QGIS (pydantic, numpy, scipy, Pillow,
   shapely, pyproj, osgeo), others from the venv (fastapi, rasterio, torch, faiss, cv2, sklearn, open_clip, mgrs, psutil).
   `main.py` registers QGIS DLL dirs and sets `PROJ_DATA`; **import `main` (or set PROJ_DATA) before using pyproj**, or you
   get "Valid PROJ data directory not found".
5. **Pydantic's regex engine** rejects large classes: `^[\w\-.]{1,300}$` fails at import ("Compiled regex exceeds size
   limit"). `SCENE_ID_RE` uses `{1,200}`; the tile-id pattern uses an ASCII class.
6. **titiler tilejson is cached by browsers** (max-age 3600). `main.py` adds `Cache-Control: no-cache`; a stale map after a
   backend fix can still be a cached tilejson.
7. **Synthetic 4-band uint16 test COGs fail titiler PNG encoding** (no rescale/band select). Real imagery renders; the
   display COG for `.SAFE` scenes is an RGB composite.
8. **Memory.** ~3 GB free RAM was the working budget on the dev machine. All raster work is windowed in 512-row strips
   (`STRIP_ROWS`); the full-scale real pair peaks at **~3.7 GB RSS** (labels raster int32 = 480 MB, plus per-level masks in
   grouping). Do not add full-size float arrays.
9. **Vite dev server re-optimises dependencies after `npm install`** and reloads the page once; a browser test started
   immediately after installing a package can fail with "detached Frame". Load the page once, then run the test.
10. **Frontend hostnames are mixed:** `api.js` uses `127.0.0.1:8000`, `CompareMap.jsx`/`MapView.jsx` hard-code
    `localhost:8000`. CORS allows `localhost:5173` and `127.0.0.1:5173`. Works today; unify if it ever bites.
11. **Nested buttons are invalid HTML.** Result cards are `div role="button"` because they contain a Find Similar button.
12. **Agent tooling gotchas** (only if an AI agent continues): the Bash tool breaks on large heredocs with quotes and
    `sed` with nested quoting — write Python patch scripts to a file and run them; MSYS rewrites `/c/...` arguments; in
    patch scripts remember Windows backslashes in regexes.

---

## 5. Dependencies

`requirements.txt` (root and `backend/`, identical) **pins versions that are not what is installed and tested.** Installed
(2026-09-21): fastapi 0.141.1, uvicorn 0.53.0, pydantic 2.13.4, rasterio 1.5.1, GDAL 3.13.3 (osgeo, from QGIS), numpy
2.4.6, scipy 1.18.0, scikit-learn 1.9.1, opencv-python-headless 5.0.0.93, faiss-cpu 1.15.1, titiler.core/application 2.3.0,
open_clip_torch 3.3.0, torch 2.14.0, Pillow 12.3.0, huggingface_hub 1.32.0, mgrs 1.5.4, psutil 7.2.2, pytest 9.1.1,
httpx 0.28.1. `scikit-image` is pinned but **not installed** and unused. `shapely` is installed (via QGIS) but unused and
not listed. **Action:** regenerate the file from `pip freeze` of the working environment, since AGENTS.md calls
reproducibility a submission requirement.

Added this milestone: `mgrs==1.5.4` (pure Python), `psutil==7.2.2` (hardware info), and on the frontend `mgrs@^2.2.0`
(`npm install mgrs`; npm prints an esbuild "allow-scripts" warning that is harmless). Python and JS MGRS output were
checked to match (`43RCL0333298814` for lat 28, lon 73).

Model weights: `checkpoints/RemoteCLIP-ViT-B-32.pt` must be present; the app never downloads anything. The log line
"No pretrained weights loaded for model 'ViT-B-32'" comes from `open_clip.create_model_and_transforms` before the local
checkpoint is loaded and is expected.

---

## 6. Bugs found and how they were fixed (chronological, most useful ones)

| Problem | Cause | Fix |
|---|---|---|
| Search results stale after importing another scene | search state kept the old scene's results | `resetSearch()` on scene change; silent re-run when the pair signature changes |
| PIF normalisation biased | pixel selection depended on the very difference being tested | selection-free starting fit, then residual-based re-selection (`PIF_ITERATIONS`) |
| "File not found" on nested `.SAFE` | MAX_PATH | `io_path` / `plain_path` |
| OpenCV error text leaked file paths to the UI | raw exception messages | `pipeline._safe_message` strips paths |
| Test fixtures had corrupted SCL classes | JP2OpenJPEG default is lossy | write JP2 with `REVERSIBLE=YES`, fall back to GTiff |
| Confidence saturated near 1.0 | percentile ranked vs pixels | rank vs blobs, later vs per-detection peaks |
| Confirm/reject didn't refresh the list | search-result list held its own copy of candidates | `setReviewStatus` writes browse list, search list and detail; optimistic with rollback (13 ms vs ~800 ms) |
| "Construction" never fired | rule required more bare pixels at B than A (impossible for bare→bare) | removed that condition |
| Test "built" surface classed as water | NDWI>0 when NIR < green | fixture NIR set above green; limit recorded in §9 |
| Stored MGRS off by 1 m vs stored centroid | MGRS from unrounded coordinates | derive from the rounded stored centroid |
| Trace showed tile `43PHM` | pipeline stores it without the `T` | prefix `T` for display |
| `/api/changes/export` → 422 | route order vs `{candidate_id:int}` | moved to `/api/export/changes` |
| App wouldn't import: regex size | pydantic `\w{1,300}` | ASCII class |
| Merged confidence scale compressed (weakest = 0.79) | ranking all fragments | rank per-detection peaks |
| **Merge produced 151,937 ha / 94,522 ha detections on the real pair** | transitive chaining on a dense scene (the 10 px pass too) | size guard at every level + fallback ladder (§3) |
| Storage cap dropped the largest changes | cap by confidence only | `STORED_AREA_RESERVE` |
| "Sort by area" could miss the largest | cap applied by confidence before sorting | cap follows the sort |
| Find Similar "does nothing" | errors rendered only inside the collapsed Semantic Results section; likely stale backend 404 | section opens on click, spinner text, titled error, 404 hint to restart |
| Grouping tests broke fixtures | patches 40 px apart now merge | `test_change_filters.py` disables the wide pass via an autouse fixture |

---

## 7. How to verify each feature

Start a backend from `backend/` (or a scratch dir, see below), then:

- **Ingest + pipeline:** import a `.SAFE` folder in the UI (Browse → paste path); the Workspace panel shows steps;
  `GET /api/catalog/status`; `GET /api/changes/jobs`.
- **Change detection:** `GET /api/changes` → candidates, `pairs`, `area`; `GET /api/changes/{id}` → `display_bounds`,
  `processing_details`, `direction_evidence`, `mgrs`.
- **Direction:** synthetic transitions in `tests/test_direction.py` (veg→bare clearance, water→bare, bare→veg, bare→built).
- **Merging:** `tests/test_grouping.py` (pure rules incl. size guard; scattered patches → one detection; area sort; cap).
  On real data, re-run the pair (below) and read `job.details.grouping` (`blobs`, `groups`, `large_area_groups`,
  `too_large_to_merge`, `largest_group_blobs`).
- **Search + boost:** `POST /api/search {"query":"new construction","top_k":10}` → `change_meta.direction_hints`,
  `direction_boost`; tests in `test_direction.py`, `test_unified_search.py`.
- **Find Similar:** `POST /api/search/similar {"tile_id":"<id>"}` or `{"candidate_id":N}` → `label`, 10 results, seed
  excluded; `tests/test_demo_features.py`. A 404 `{"detail":"Not Found"}` means the backend is stale: restart it.
- **MGRS:** tile/candidate/search results carry `mgrs`; status bar shows it on map hover; `test_grouping.py::…mgrs…`.
- **Export:** `curl localhost:8000/api/export/changes -o out.geojson` (EPSG:4326 `crs`, one feature per candidate).
- **Negative evidence:** a pair with no candidates (or a search with `change_status="no_change_detected"`) shows the card;
  Export downloads the JSON (`area_bounds`, `mgrs`, `dates_analysed`, `valid_coverage`, `confidence_in_absence`, `statement`).
- **Eval manifest:** `GET /api/eval/manifest`; also `backend/data/eval_manifest.json`; UI: Export ▸ Evaluation manifest.
- **Decision trace / wider view:** open a candidate → "Processing Details" accordion; the map opens on a 4× window.

**Browser tests** were done with `puppeteer-core` driving system Chrome
(`C:\Program Files\Google\Chrome\Application\chrome.exe`, headless, swiftshader flags) against a scratch backend + Vite;
those scripts lived in the assistant's temp scratchpad and are **not in the repo**. Recreate the pattern: fresh
`ui_run/data` directory as the backend's cwd, `uvicorn main:app --app-dir <repo>/backend`, seed scenes with
`tests/s2_factory.py`, import through the UI's `prompt()` (accept the dialog with the path), assert on `data-testid`s
(`change-card`, `change-title`, `direction-icon`, `change-mgrs`, `semantic-card`, `find-similar-tile`,
`find-similar-change`, `similar-label`, `export-button`, `export-geojson`, `export-manifest`, `processing-toggle`,
`trace-*`, `status-mgrs`, `negative-evidence`, `negative-mgrs`, `search-error-title`). Set downloads with
`Page.setDownloadBehavior`. **Never point a test at `backend/data`.**

**Testing against the real catalog safely:** copy `backend/data/iris_catalog.db*` and `faiss_index.bin` into a scratch
`data/`, junction (`mklink /J`) `cogs` and `crops` to the real folders (read-only use), and run uvicorn with that cwd.
**Re-running a pair:** in that copy, delete the rows in `reviews`, `change_candidates` and `jobs`, then call
`change_detection.run_change_detection(scene_a_id, scene_b_id)` (~206 s; radiometry 92 s, detection 44 s, grouping 10 s;
~3.7 GB RSS). Jobs in a final state are otherwise never re-run (idempotent by design). Doing this on the real
`backend/data` will replace the user's candidates and reviews: ask first.

---

## 8. Test infrastructure notes

`tests/s2_factory.py::Scenario` builds a 512×512 (default) synthetic scene with real `.SAFE` layout: `patches`
(NIR-factor changes, unclassifiable by design), `landcover=[(r0,r1,c0,c1,kind)]` with kinds `vegetation|bare|built|water`
at fixed reflectances (`LANDCOVER_REFLECTANCE`; NIR values are chosen so each transition is a clear B08 change), `shift`,
`gain`, `offset`, `cloud`, `nodata_rows`, `origin`. Two scenes with the same tile ID skip alignment; use a different
tile name to exercise ECC/ORB. A weak change (e.g. bare→built, ~800 DN) next to strong ones is not detected because the
K-means threshold rises: keep fixture changes comparable in NIR magnitude.

---

## 9. Known limitations and what to build next

- **K-means threshold** is the midpoint of two centroids of |ΔNIR|; weak real changes are lost beside strong ones.
- **NDWI>0 water rule** classes bright low-NIR surfaces (some built-up) as water. No SWIR → no NDBI.
- **Dense scenes:** the real pair yields ~15.6k detections; only 5,000 are stored (1,000s dropped by confidence). Consider
  raising `MAX_STORED_CANDIDATES`, or a min-area/min-confidence display default. Most are 1–5 ha field-scale changes; the
  Jewar airport change was not identified in the data (the largest item is a single 9,894 ha unmerged blob).
- **Merge limits** (5 km / 2,500 ha) are guesses; calibrate them on the real pair with the user.
- **Not built** (per `architecture.md`): persistence/seasonal filter, linear-feature shape descriptors, DEM terrain term,
  archive-wide clustering, GAE heatmaps, Visual Evidence Card (printable HTML), feedback reranking, calibration against OSCD.
- Weights, thresholds and NDVI/NDWI cut-offs are unvalidated starting values (see `params.py` header).
- Existing user catalog needs a re-run to get merged detections, geometry outlines and MGRS on candidates (tiles are
  backfilled automatically by the schema migration).
- Housekeeping: commit the work; add `backend/data/` to `.gitignore`; regenerate requirements; consider removing the stray
  root `main.py`/`database.py`/`main.js`/`preload.js`/`package.json`; unify `localhost` vs `127.0.0.1` in the frontend;
  add the browser test scripts to the repo (e.g. `frontend/e2e/`).
- Hard rules still apply (AGENTS.md): no network except localhost, no auth, bind 127.0.0.1, Electron
  `contextIsolation/nodeIntegration/sandbox` untouched, SQLite WAL + `BEGIN IMMEDIATE`, module boundaries.
