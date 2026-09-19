import React, { useEffect, useRef, useImperativeHandle, forwardRef } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

const MapView = forwardRef(function MapView(
  { currentScene, onMouseMove, onMapReady },
  ref
) {
  const mapContainer = useRef(null);
  const mapInstance = useRef(null);
  const activeBounds = useRef(null);

  // Expose zoomIn, zoomOut, fitBounds via ref
  useImperativeHandle(ref, () => ({
    zoomIn: () => {
      if (mapInstance.current) mapInstance.current.zoomIn({ duration: 300 });
    },
    zoomOut: () => {
      if (mapInstance.current) mapInstance.current.zoomOut({ duration: 300 });
    },
    fitBounds: () => {
      if (mapInstance.current && activeBounds.current) {
        mapInstance.current.fitBounds(activeBounds.current, {
          padding: 40,
          duration: 1000,
          maxZoom: 16,
        });
      }
    },
  }));

  // Initialize MapLibre GL map
  useEffect(() => {
    if (!mapContainer.current || mapInstance.current) return;

    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {},
        layers: [
          {
            id: "background",
            type: "background",
            paint: {
              "background-color": "#18181b", // Dark neutral canvas
            },
          },
        ],
      },
      center: [78.9629, 20.5937], // Centered over India by default
      zoom: 4,
      attributionControl: false,
    });

    map.on("load", () => {
      mapInstance.current = map;
      if (onMapReady) onMapReady(map);
    });

    map.on("mousemove", (e) => {
      if (onMouseMove) {
        onMouseMove({
          lng: e.lngLat.lng,
          lat: e.lngLat.lat,
          zoom: map.getZoom(),
          bearing: map.getBearing(),
        });
      }
    });

    return () => {
      map.remove();
      mapInstance.current = null;
    };
  }, []);

  // Update raster layer when currentScene changes
  useEffect(() => {
    const map = mapInstance.current;
    if (!map || !currentScene || (!currentScene.cog_path && !currentScene.cog_url)) return;

    const targetUrl = currentScene.cog_url || currentScene.cog_path;
    const sourceId = `cog-source-${currentScene.scene_id}`;
    const layerId = `cog-layer-${currentScene.scene_id}`;

    // Remove any previous COG layers & sources
    const layers = map.getStyle().layers || [];
    layers.forEach((l) => {
      if (l.id.startsWith("cog-layer-")) {
        if (map.getLayer(l.id)) map.removeLayer(l.id);
      }
    });
    const sources = Object.keys(map.getStyle().sources || {});
    sources.forEach((s) => {
      if (s.startsWith("cog-source-")) {
        if (map.getSource(s)) map.removeSource(s);
      }
    });

    async function loadCog() {
      try {
        const tileJsonUrl = `http://localhost:8000/tiles/cog/tilejson.json?url=${encodeURIComponent(
          targetUrl
        )}`;

        // MapLibre natively understands TileJSON via url property
        map.addSource(sourceId, {
          type: "raster",
          url: tileJsonUrl,
          tileSize: 256,
        });

        // Add raster layer with bilinear resampling
        map.addLayer({
          id: layerId,
          type: "raster",
          source: sourceId,
          paint: {
            "raster-resampling": "linear",
            "raster-fade-duration": 200,
          },
        });

        // Use bounds directly from tilejson or bounds_wgs84
        if (currentScene.bounds_wgs84 && currentScene.bounds_wgs84.length === 4) {
          const fitTarget = [
            [currentScene.bounds_wgs84[0], currentScene.bounds_wgs84[1]],
            [currentScene.bounds_wgs84[2], currentScene.bounds_wgs84[3]],
          ];
          activeBounds.current = fitTarget;
          map.fitBounds(fitTarget, {
            padding: 40,
            duration: 1000,
            maxZoom: 16,
          });
        } else {
          // Fetch TileJSON to get WGS84 bounds directly
          fetch(tileJsonUrl)
            .then((res) => res.json())
            .then((tileJson) => {
              if (tileJson.bounds && tileJson.bounds.length === 4) {
                const fitTarget = [
                  [tileJson.bounds[0], tileJson.bounds[1]],
                  [tileJson.bounds[2], tileJson.bounds[3]],
                ];
                activeBounds.current = fitTarget;
                map.fitBounds(fitTarget, {
                  padding: 40,
                  duration: 1000,
                  maxZoom: 16,
                });
              }
            })
            .catch((err) => console.warn("TileJSON bounds lookup error:", err));
        }
      } catch (e) {
        console.error("Error loading COG layer into MapLibre:", e);
      }
    }

    if (map.isStyleLoaded()) {
      loadCog();
    } else {
      map.once("styledata", loadCog);
    }
  }, [currentScene]);

  return (
    <div className="relative w-full h-full flex-1 overflow-hidden bg-[#18181b]">
      <div ref={mapContainer} className="w-full h-full" />
    </div>
  );
});

export default MapView;
