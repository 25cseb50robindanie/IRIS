import React, { useState, useRef, useEffect, useCallback } from "react";
import Toolbar from "./components/Toolbar";
import MapView from "./components/MapView";
import Sidebar from "./components/Sidebar";
import SearchBar from "./components/SearchBar";
import StatusBar from "./components/StatusBar";

const API_BASE = "http://127.0.0.1:8000";
const EMBED_TERMINAL_STATES = new Set(["ready", "failed", "not_embedded"]);
const EMBED_POLL_MS = 750;
const EMBED_MAX_POLL_FAILURES = 5;
const EMBED_ACTIVE_STATES = new Set(["queued", "tiling", "embedding", "indexing"]);
const CATALOG_RETRY_MS = 1000;
const CATALOG_MAX_ATTEMPTS = 10;

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
  const [error, setError] = useState(null);
  const [mousePos, setMousePos] = useState({ lng: undefined, lat: undefined, zoom: 4, bearing: 0 });

  // Semantic Search State
  const [searchState, setSearchState] = useState({
    query: "",
    isSearching: false,
    results: [],
    error: null,
    selectedResult: null,
  });

  const handleBrowse = async () => {
    try {
      let selectedPath = null;
      if (window.iris && typeof window.iris.openFile === "function") {
        selectedPath = await window.iris.openFile();
      } else {
        // Fallback for browser dev mode
        selectedPath = prompt("Enter full absolute file path to satellite imagery (e.g. GeoTIFF / JP2):");
      }

      if (!selectedPath) return;

      const fileName = selectedPath.split(/[\/\\]/).pop();
      setIngestState({ status: "ingesting", fileName });
      setError(null);

      const response = await fetch(`${API_BASE}/api/ingest`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ file_path: selectedPath }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: HTTP ${response.status}`);
      }

      // The scene is viewable now; embedding continues in the background and is polled below
      const data = await response.json();
      setCurrentScene(data);
      setEmbedStatus({ scene_id: data.scene_id, state: data.embedding_status || "queued" });
      setIngestState({ status: "ready", fileName });
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
  }, [refreshCatalog]);

  // Once a scene finishes embedding, its vectors are searchable: refresh the counts that gate search
  const embedState = embedStatus?.state;
  useEffect(() => {
    if (embedState === "ready") {
      refreshCatalog().catch((err) => console.warn("Catalog refresh failed:", err));
    }
  }, [embedState, refreshCatalog]);

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

  const handleSearch = async (queryText) => {
    if (!queryText.trim() || searchDisabledReason) return;

    setSearchState((prev) => ({
      ...prev,
      query: queryText,
      isSearching: true,
      error: null,
      selectedResult: null,
    }));

    try {
      const response = await fetch(`${API_BASE}/api/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: queryText, top_k: 10 }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Search failed: HTTP ${response.status}`);
      }

      const data = await response.json();
      const results = data.results || [];

      setSearchState((prev) => ({
        ...prev,
        isSearching: false,
        results: results,
        selectedResult: results.length > 0 ? results[0] : null,
      }));

      // Automatically fly to top result if available
      if (results.length > 0 && mapRef.current) {
        mapRef.current.flyToBounds(results[0].bounds);
      }
    } catch (err) {
      console.error("Semantic search failed:", err);
      setSearchState((prev) => ({
        ...prev,
        isSearching: false,
        results: [],
        error: err.message || "Failed to execute search",
      }));
    }
  };

  const handleSelectResult = (result) => {
    setSearchState((prev) => ({
      ...prev,
      selectedResult: result,
    }));
    if (mapRef.current && result.bounds) {
      mapRef.current.flyToBounds(result.bounds);
    }
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
      />

      {/* 2. Main Workspace: Map Canvas + Right Sidebar */}
      <div className="flex flex-1 overflow-hidden relative">
        <MapView
          ref={mapRef}
          currentScene={currentScene}
          selectedResult={searchState.selectedResult}
          onMouseMove={setMousePos}
        />
        <Sidebar
          ingestState={ingestState}
          currentScene={currentScene}
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
