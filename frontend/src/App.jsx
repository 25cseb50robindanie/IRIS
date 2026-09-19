import React, { useState, useRef } from "react";
import Toolbar from "./components/Toolbar";
import MapView from "./components/MapView";
import Sidebar from "./components/Sidebar";
import StatusBar from "./components/StatusBar";

export default function App() {
  const mapRef = useRef(null);
  const [ingestState, setIngestState] = useState({
    status: "idle", // 'idle' | 'ingesting' | 'ready' | 'error'
    fileName: "",
  });
  const [currentScene, setCurrentScene] = useState(null);
  const [error, setError] = useState(null);
  const [mousePos, setMousePos] = useState({ lng: undefined, lat: undefined, zoom: 4, bearing: 0 });

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

      const response = await fetch("http://127.0.0.1:8000/api/ingest", {
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

      const data = await response.json();
      setCurrentScene(data);
      setIngestState({ status: "ready", fileName });
    } catch (err) {
      console.error("Ingestion failed:", err);
      setError(err.message || "Failed to ingest image");
      setIngestState({ status: "error", fileName: "" });
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
          onMouseMove={setMousePos}
        />
        <Sidebar
          ingestState={ingestState}
          currentScene={currentScene}
          error={error}
        />
      </div>

      {/* 3. Bottom Status Bar */}
      <StatusBar
        mousePos={mousePos}
        activeCrs={currentScene?.crs}
      />
    </div>
  );
}
