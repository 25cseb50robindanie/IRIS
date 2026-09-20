import React, { useState, useRef, useEffect, useCallback } from "react";
import Toolbar from "./components/Toolbar";
import MapView from "./components/MapView";
import Sidebar from "./components/Sidebar";
import SearchBar from "./components/SearchBar";
import StatusBar from "./components/StatusBar";
import ChangeComparison from "./components/ChangeComparison";
import { DEFAULT_FILTERS } from "./components/ChangeResults";

const API_BASE = "http://127.0.0.1:8000";
const EMBED_TERMINAL_STATES = new Set(["ready", "failed", "not_embedded"]);
const EMBED_POLL_MS = 750;
const EMBED_MAX_POLL_FAILURES = 5;
const EMBED_ACTIVE_STATES = new Set(["queued", "tiling", "embedding", "indexing"]);
const CATALOG_RETRY_MS = 1000;
const CATALOG_MAX_ATTEMPTS = 10;
const CHANGES_POLL_MS = 4000;
const PIPELINE_POLL_MS = 1000;
const PIPELINE_MAX_MISSES = 5;
const FILTER_DEBOUNCE_MS = 200;

// The analyst's change filters as query parameters. Filtering, the per-pair cap and counts are the server's job.
function changesQuery(f) {
  const q = new URLSearchParams();
  q.set("min_confidence", String(f.minConfidence));
  for (const group of ["types", "directions"]) {
    const on = Object.entries(f[group]).filter(([, v]) => v).map(([k]) => k);
    if (on.length < Object.keys(f[group]).length) q.set(group, on.join(","));
  }
  q.set("sort", f.sort === "match" ? "confidence" : f.sort); // "best match" only exists for search results
  if (f.jobId != null) q.set("job_id", String(f.jobId));
  return q.toString();
}

// Chosen by the client so progress can be polled while the (synchronous) ingest request is still running
function newPipelineId() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID().replace(/-/g, "");
  return `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`.padEnd(16, "0");
}

export default function App() {
  const mapRef = useRef(null);
  const [ingestState, setIngestState] = useState({
    status: "idle", // 'idle' | 'ingesting' | 'ready' | 'error'
    fileName: "",
  });
  const [currentScene, setCurrentScene] = useState(null);
  // Background embedding progress for the current scene (GET /api/status/{scene_id})
  const [embedStatus, setEmbedStatus] = useState(null);
  // Existing catalog state from GET /api/catalog/status (null until the first response)
  const [catalog, setCatalog] = useState(null);
  const [catalogError, setCatalogError] = useState(null);

  // Change detection: candidates, the pair jobs behind them, and the before/after view for one candidate
  const [changes, setChanges] = useState({ total: 0, matching: 0, shown: 0, candidates: [], pairs: [] });
  const [changeJobs, setChangeJobs] = useState([]);
  const [changeFilters, setChangeFilters] = useState(DEFAULT_FILTERS);
  const filtersRef = useRef(DEFAULT_FILTERS);
  filtersRef.current = changeFilters;
  // "search": the changes that match the last query; "all": every detected change (filtered)
  const [changeView, setChangeView] = useState("all");
  const [selectedChangeId, setSelectedChangeId] = useState(null);
  const [changeDetail, setChangeDetail] = useState(null);
  const [changeLoading, setChangeLoading] = useState(false);
  const [changeError, setChangeError] = useState(null);
  const [error, setError] = useState(null);
  const [mousePos, setMousePos] = useState({ lng: undefined, lat: undefined, zoom: 4, bearing: 0 });

  // Semantic Search State
  const [searchState, setSearchState] = useState({
    query: "",
    isSearching: false,
    results: [], // semantic tiles
    changeResults: [], // changes whose after-image matches the query
    changeStatus: null, // found | no_match | no_change_detected | no_comparison_available
    changeMeta: null,
    hasSearched: false,
    searchedScene: null,
    error: null,
    selectedResult: null,
  });
  const searchStateRef = useRef(null);
  searchStateRef.current = searchState;
  const [searchCount, setSearchCount] = useState(0); // bumps per search so the Workspace re-applies its defaults

  // Imported scenes (toolbar switcher) and the step-by-step progress of the import in flight
  const [scenes, setScenes] = useState([]);
  const [pipelineId, setPipelineId] = useState(null);
  const [pipeline, setPipeline] = useState(null);
  // Latest values for async handlers that outlive a render
  const currentSceneRef = useRef(null);
  currentSceneRef.current = currentScene;
  const scenesRef = useRef([]);
  scenesRef.current = scenes;

  const handleBrowse = async () => {
    try {
      let selectedPath = null;
      if (window.iris && typeof window.iris.openFile === "function") {
        selectedPath = await window.iris.openFile();
      } else {
        // Fallback for browser dev mode
        selectedPath = prompt("Enter the full path to a satellite image (GeoTIFF / JP2) or a Sentinel-2 .SAFE folder:");
      }

      if (!selectedPath) return;

      const fileName = selectedPath.split(/[\/\\]/).pop();
      setIngestState({ status: "ingesting", fileName });
      setError(null);
      const pid = newPipelineId();
      setPipeline(null);
      setPipelineId(pid);

      const response = await fetch(`${API_BASE}/api/ingest`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ file_path: selectedPath, pipeline_id: pid }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: HTTP ${response.status}`);
      }

      // The scene is viewable now; embedding continues in the background and is polled below
      const data = await response.json();
      resetSearch(); // the imported scene is now the one on the map, so earlier results (and their change status) are stale
      setCurrentScene(data);
      setEmbedStatus({ scene_id: data.scene_id, state: data.embedding_status || "queued" });
      setIngestState({ status: "ready", fileName });
      refreshScenes().catch(() => {});
    } catch (err) {
      console.error("Ingestion failed:", err);
      setError(err.message || "Failed to ingest image");
      setIngestState({ status: "error", fileName: "" });
    }
  };

  const refreshCatalog = useCallback(async () => {
    const res = await fetch(`${API_BASE}/api/catalog/status`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    setCatalog(data);
    return data;
  }, []);

  const refreshScenes = useCallback(async () => {
    const res = await fetch(`${API_BASE}/api/scenes`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const list = (await res.json()).scenes;
    setScenes((prev) => (JSON.stringify(prev) === JSON.stringify(list) ? prev : list));
    return list;
  }, []);

  // On startup, load what is already imported and resume on the latest scene
  useEffect(() => {
    let cancelled = false;
    let timer = null;
    let attempts = 0;

    const load = async () => {
      try {
        const data = await refreshCatalog();
        if (cancelled) return;
        setCatalogError(null);
        refreshScenes().catch(() => {});
        const latest = data.latest_scene;
        if (latest) {
          // Never replace a scene the analyst imported while this request was in flight
          setCurrentScene((prev) => prev ?? latest);
          setIngestState((prev) =>
            prev.status === "idle" ? { status: "ready", fileName: latest.scene_id } : prev
          );
        }
      } catch (err) {
        if (cancelled) return;
        attempts += 1;
        if (attempts >= CATALOG_MAX_ATTEMPTS) {
          setCatalogError("Could not reach the IRIS backend.");
          return;
        }
        timer = setTimeout(load, CATALOG_RETRY_MS);
      }
    };

    load();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [refreshCatalog, refreshScenes]);

  // Keep the change-detection lists current. Jobs run in the background after ingest, so this polls; a state is
  // only replaced when it actually changed, so an idle app does not re-render.
  const refreshChanges = useCallback(async () => {
    const [listRes, jobsRes] = await Promise.all([
      fetch(`${API_BASE}/api/changes?${changesQuery(filtersRef.current)}`),
      fetch(`${API_BASE}/api/changes/jobs`),
    ]);
    if (!listRes.ok || !jobsRes.ok) throw new Error("Change status unavailable");
    const list = await listRes.json();
    const jobs = (await jobsRes.json()).jobs;
    setChanges((prev) => (JSON.stringify(prev) === JSON.stringify(list) ? prev : list));
    setChangeJobs((prev) => (JSON.stringify(prev) === JSON.stringify(jobs) ? prev : jobs));
  }, []);

  useEffect(() => {
    let cancelled = false;
    const tick = () => {
      if (cancelled || document.hidden) return;
      refreshChanges().catch(() => {});
    };
    tick();
    const timer = setInterval(tick, CHANGES_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [refreshChanges]);

  // Re-query when the analyst moves a filter (debounced so dragging the slider is one request, not fifty)
  useEffect(() => {
    const t = setTimeout(() => refreshChanges().catch(() => {}), FILTER_DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [changeFilters, refreshChanges]);

  // A selected pair that no longer exists (its scene was deleted) must not leave the list filtered to nothing
  useEffect(() => {
    if (changeFilters.jobId != null && changes.pairs.length && !changes.pairs.some((p) => p.job_id === changeFilters.jobId)) {
      setChangeFilters((f) => ({ ...f, jobId: null }));
    }
  }, [changes.pairs, changeFilters.jobId]);

  // A comparison can finish after a search was run (change detection runs in the background). The change half of
  // the results then describes a state that no longer exists, so it is re-run in place.
  const pairSignature = changes.pairs.filter((p) => p.status === "completed").map((p) => `${p.job_id}:${p.candidates}`).join(",");
  const lastPairSignature = useRef(pairSignature);
  const rerunSearch = useRef(null);
  useEffect(() => {
    if (lastPairSignature.current === pairSignature) return;
    lastPairSignature.current = pairSignature;
    if (searchStateRef.current.hasSearched && rerunSearch.current) rerunSearch.current();
  }, [pairSignature]);

  const changeViewTo = (view) => {
    setChangeView(view);
    // "best match" only exists for search results; the full list is ordered by what the server can sort on
    setChangeFilters((f) => (view === "all" && f.sort === "match" ? { ...f, sort: "confidence" } : f));
  };

  const loadChangeDetail = useCallback(async (id) => {
    const res = await fetch(`${API_BASE}/api/changes/${id}`);
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Could not load candidate (HTTP ${res.status})`);
    }
    return res.json();
  }, []);

  const openChange = async (id) => {
    setSelectedChangeId(id);
    setChangeDetail(null);
    setChangeError(null);
    setChangeLoading(true);
    try {
      setChangeDetail(await loadChangeDetail(id));
    } catch (err) {
      setChangeError(err.message || "Could not load candidate");
    } finally {
      setChangeLoading(false);
    }
  };

  const closeChange = () => {
    setSelectedChangeId(null);
    setChangeDetail(null);
    setChangeError(null);
  };

  // The candidate is held in three places (the browse list, the search-result list and the open detail), so a
  // decision has to be written to all of them
  const setReviewStatus = (id, status) => {
    const mark = (c) => (c.candidate_id === id ? { ...c, review_status: status } : c);
    setChanges((prev) => ({ ...prev, candidates: prev.candidates.map(mark) }));
    setSearchState((prev) => ({ ...prev, changeResults: prev.changeResults.map(mark) }));
    setChangeDetail((prev) => (prev && prev.candidate_id === id ? { ...prev, review_status: status } : prev));
  };

  // Confirm / reject: persisted in the reviews table. The lists change the moment the analyst decides and are put
  // back if the save fails; the server's copy is fetched afterwards to settle any difference.
  const reviewChange = async (id, decision) => {
    const previous = changeDetail?.candidate_id === id ? changeDetail.review_status : "pending";
    setReviewStatus(id, decision);
    try {
      const res = await fetch(`${API_BASE}/api/changes/${id}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Could not save decision (HTTP ${res.status})`);
      }
    } catch (err) {
      setReviewStatus(id, previous);
      throw err;
    }
    try {
      setChangeDetail(await loadChangeDetail(id));
    } catch (err) {
      console.warn("Could not reload the reviewed candidate:", err);
    }
    refreshChanges().catch(() => {});
  };

  // Once a scene finishes embedding, its vectors are searchable: refresh the counts that gate search
  const embedState = embedStatus?.state;
  useEffect(() => {
    if (embedState === "ready") {
      refreshCatalog().catch((err) => console.warn("Catalog refresh failed:", err));
      refreshScenes().catch(() => {});
    }
  }, [embedState, refreshCatalog, refreshScenes]);

  // Step-by-step import progress. It starts polling before the ingest request returns and carries on through the
  // background embedding and change-detection job, until the pipeline reports done or failed.
  useEffect(() => {
    if (!pipelineId) return undefined;
    let cancelled = false;
    let timer = null;
    let misses = 0;

    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/pipeline/${encodeURIComponent(pipelineId)}`);
        if (res.status === 404) {
          misses += 1; // the ingest request may not have registered the pipeline yet
          if (misses >= PIPELINE_MAX_MISSES) return;
        } else if (res.ok) {
          misses = 0;
          const data = await res.json();
          if (cancelled) return;
          setPipeline(data);
          if (data.state !== "running") {
            refreshCatalog().catch(() => {});
            refreshScenes().catch(() => {});
            refreshChanges().catch(() => {});
            return;
          }
        }
      } catch (err) {
        if (cancelled) return;
      }
      timer = setTimeout(poll, PIPELINE_POLL_MS);
    };

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [pipelineId, refreshCatalog, refreshScenes, refreshChanges]);

  const dismissPipeline = () => {
    setPipeline(null);
    setPipelineId(null);
  };

  const showScene = (scene) => {
    // A finished import summary belongs to the scene it imported; don't leave it up over a different scene
    setPipeline((prev) => (prev && prev.state === "running" ? prev : null));
    setCurrentScene(scene);
    setIngestState({ status: "ready", fileName: scene.scene_id });
    setEmbedStatus(null); // the status poll for the new scene fills this in
  };

  // Search results are a snapshot of one scene at one moment; when the scene on the map changes they no longer apply
  const resetSearch = () => {
    setSearchState((prev) => ({
      ...prev,
      results: [],
      changeResults: [],
      changeStatus: null,
      changeMeta: null,
      hasSearched: false,
      selectedResult: null,
    }));
    setChangeView("all");
  };

  // Toolbar: switch the map to another imported scene
  const switchToScene = (sceneId) => {
    const scene = scenesRef.current.find((s) => s.scene_id === sceneId);
    if (!scene || !scene.cog_available) return;
    resetSearch(); // results belong to the scene they were searched in; a different scene starts clean
    closeChange();
    showScene(scene);
  };

  // Toolbar: delete a scene (vectors, catalog rows, files), then move on to the next available one
  const deleteScene = async (sceneId) => {
    const res = await fetch(`${API_BASE}/api/scenes/${encodeURIComponent(sceneId)}`, { method: "DELETE" });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Could not delete the scene (HTTP ${res.status})`);
    }
    const result = await res.json();
    const remaining = await refreshScenes();
    refreshCatalog().catch(() => {});
    refreshChanges().catch(() => {});

    // Drop anything on screen that pointed at the deleted scene
    setSearchState((prev) => ({
      ...prev,
      results: prev.results.filter((r) => r.scene_id !== sceneId),
      changeResults: prev.changeResults.filter((r) => r.scene_a_id !== sceneId && r.scene_b_id !== sceneId),
      selectedResult: prev.selectedResult?.scene_id === sceneId ? null : prev.selectedResult,
    }));
    if (changeDetail?.scene_a?.scene_id === sceneId || changeDetail?.scene_b?.scene_id === sceneId) closeChange();

    if (currentSceneRef.current?.scene_id === sceneId) {
      const next = remaining.find((s) => s.cog_available);
      if (next) {
        showScene(next);
      } else {
        setCurrentScene(null);
        setIngestState({ status: "idle", fileName: "" });
        setEmbedStatus(null);
      }
    }
    return result;
  };

  // Poll embedding progress until the job reaches a terminal state
  const sceneId = currentScene?.scene_id;
  useEffect(() => {
    if (!sceneId) return undefined;

    let cancelled = false;
    let timer = null;
    let failures = 0;

    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/status/${encodeURIComponent(sceneId)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (cancelled) return;
        failures = 0;
        setEmbedStatus(data);
        if (EMBED_TERMINAL_STATES.has(data.state)) return;
      } catch (err) {
        if (cancelled) return;
        failures += 1;
        if (failures >= EMBED_MAX_POLL_FAILURES) {
          setEmbedStatus({
            scene_id: sceneId,
            state: "failed",
            error: "Lost connection to the IRIS backend while checking embedding progress.",
          });
          return;
        }
      }
      timer = setTimeout(poll, EMBED_POLL_MS);
    };

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [sceneId]);

  // Why search is unavailable right now, or null when it can run. The backend is never called on an empty index.
  let searchDisabledReason = null;
  if (ingestState.status === "ingesting") {
    searchDisabledReason = "Importing scene...";
  } else if (!catalog) {
    searchDisabledReason = catalogError ? "Backend not reachable." : "Connecting to backend...";
  } else if (catalog.faiss_vectors === 0) {
    searchDisabledReason = EMBED_ACTIVE_STATES.has(embedState)
      ? "Embedding in progress. Search unlocks when it finishes."
      : "Import imagery to enable search.";
  }

  // silent: refresh the results in place (no spinner, no map movement, the analyst's selection and panel state stay)
  const handleSearch = async (queryText, { silent = false } = {}) => {
    if (!queryText.trim() || searchDisabledReason) return;

    if (!silent) {
      setSearchState((prev) => ({
        ...prev,
        query: queryText,
        isSearching: true,
        error: null,
        selectedResult: null,
      }));
    }
    const scopeScene = currentSceneRef.current?.scene_id || null;

    try {
      const response = await fetch(`${API_BASE}/api/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: queryText, top_k: 10, scene_id: scopeScene }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Search failed: HTTP ${response.status}`);
      }

      const data = await response.json();
      const results = data.semantic_results || [];

      setSearchState((prev) => ({
        ...prev,
        isSearching: silent ? prev.isSearching : false,
        results: results,
        changeResults: data.change_results || [],
        changeStatus: data.change_status,
        changeMeta: data.change_meta,
        hasSearched: true,
        searchedScene: scopeScene,
        selectedResult: silent ? prev.selectedResult : null,
      }));
      if (silent) return;
      setSearchCount((n) => n + 1);
      setChangeView("search");
      setChangeFilters((f) => ({ ...f, sort: "match" }));

      // Show the top result, switching the map to its scene first if it lives in another one
      if (results.length > 0) showResult(results[0]);
    } catch (err) {
      if (silent) {
        console.warn("Could not refresh search results:", err);
        return;
      }
      console.error("Semantic search failed:", err);
      setSearchState((prev) => ({
        ...prev,
        isSearching: false,
        results: [],
        changeResults: [],
        changeStatus: null,
        changeMeta: null,
        hasSearched: false,
        error: err.message || "Failed to execute search",
      }));
    }
  };

  rerunSearch.current = () => handleSearch(searchStateRef.current.query, { silent: true });

  // Select a search result. The map only has tiles for the scene it is showing, so a result from another scene
  // loads that scene first; the scene and the selection change together so the map flies straight to the result.
  const showResult = async (result) => {
    let scene = null;
    if (result.scene_id !== currentSceneRef.current?.scene_id) {
      const known = scenesRef.current.find((sc) => sc.scene_id === result.scene_id);
      scene = known || (await refreshScenes().catch(() => [])).find((sc) => sc.scene_id === result.scene_id);
      if (!scene || !scene.cog_available) {
        setSearchState((prev) => ({
          ...prev,
          error: `The scene for this result (${result.scene_id}) is no longer available.`,
        }));
        return;
      }
    }
    setSearchState((prev) => ({ ...prev, error: null, selectedResult: result }));
    if (scene) showScene(scene);
  };

  const handleSelectResult = (result) => {
    showResult(result);
  };

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-qgis-bg">
      {/* 1. Top Toolbar */}
      <Toolbar
        onBrowse={handleBrowse}
        isIngesting={ingestState.status === "ingesting"}
        onZoomIn={() => mapRef.current?.zoomIn()}
        onZoomOut={() => mapRef.current?.zoomOut()}
        onFitBounds={() => mapRef.current?.fitBounds()}
        hasActiveLayer={Boolean(currentScene)}
        scenes={scenes}
        currentSceneId={currentScene?.scene_id}
        onSelectScene={switchToScene}
        onDeleteScene={deleteScene}
        onOpenScenes={() => refreshScenes().catch(() => {})}
      />

      {/* 2. Main Workspace: Map Canvas + Right Sidebar */}
      <div className="flex flex-1 overflow-hidden relative">
        <div className="relative flex flex-1 min-w-0">
          <MapView
            ref={mapRef}
            currentScene={currentScene}
            selectedResult={searchState.selectedResult}
            onMouseMove={setMousePos}
          />
          {selectedChangeId !== null && (
            <ChangeComparison
              detail={changeDetail}
              loading={changeLoading}
              error={changeError}
              onClose={closeChange}
              onReview={reviewChange}
            />
          )}
        </div>
        <Sidebar
          ingestState={ingestState}
          currentScene={currentScene}
          scenes={scenes}
          embedStatus={embedStatus}
          catalog={catalog}
          catalogError={catalogError}
          error={error}
          searchError={searchState.error}
          searchResults={searchState.results}
          searchQuery={searchState.query}
          isSearching={searchState.isSearching}
          selectedTileId={searchState.selectedResult?.tile_id}
          onSelectResult={handleSelectResult}
          search={{
            hasSearched: searchState.hasSearched,
            query: searchState.query,
            status: searchState.changeStatus,
            results: searchState.changeResults,
            meta: searchState.changeMeta,
            sceneId: searchState.searchedScene,
          }}
          searchCount={searchCount}
          changes={changes}
          changeJobs={changeJobs}
          changeFilters={changeFilters}
          onChangeFilters={setChangeFilters}
          changeView={changeView}
          onChangeView={changeViewTo}
          selectedChangeId={selectedChangeId}
          onOpenChange={openChange}
          pipeline={pipeline}
          onDismissPipeline={dismissPipeline}
        />
      </div>

      {/* 3. Semantic Search Bar (above status bar) */}
      <SearchBar
        onSearch={handleSearch}
        isSearching={searchState.isSearching}
        disabledReason={searchDisabledReason}
      />

      {/* 4. Bottom Status Bar */}
      <StatusBar
        mousePos={mousePos}
        activeCrs={currentScene?.crs}
      />
    </div>
  );
}
