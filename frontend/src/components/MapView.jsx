import React, { useEffect, useRef, useState, useImperativeHandle, forwardRef } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

const MapView = forwardRef(function MapView(
  { currentScene, selectedResult, onMouseMove, onMapReady },
  ref
) {
  const mapContainer = useRef(null);
  const mapInstance = useRef(null);
  const activeBounds = useRef(null);
  // The scene layer can only be added once the map has loaded; a scene resumed from the catalog
  // on startup may arrive before that, so the layer effect must re-run when the map becomes ready.
  const [mapReady, setMapReady] = useState(false);

  // Expose zoomIn, zoomOut, fitBounds via ref
  useImperativeHandle(ref, () => ({
    zoomIn: () => {
      if (mapInstance.current) mapInstance.current.zoomIn({ duration: 300 });
    },
    zoomOut: () => {
      if (mapInstance.current) mapInstance.current.zoomOut({ duration: 300 });
    },
    fitBounds: (customBounds) => {
      const target = customBounds || activeBounds.current;
      if (mapInstance.current && target) {
        mapInstance.current.fitBounds(target, {
          padding: 50,
          duration: 1000,
          maxZoom: 17,
        });
      }
    },
    flyToBounds: (boundsWgs84) => {
      if (mapInstance.current && boundsWgs84 && boundsWgs84.length === 4) {
        const fitTarget = [
          [boundsWgs84[0], boundsWgs84[1]],
          [boundsWgs84[2], boundsWgs84[3]],
        ];
        mapInstance.current.fitBounds(fitTarget, {
          padding: 80,
          duration: 1200,
          maxZoom: 17,
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

      // Add GeoJSON source for selected crop highlight box
      if (!map.getSource("crop-highlight-source")) {
        map.addSource("crop-highlight-source", {
          type: "geojson",
          data: {
            type: "FeatureCollection",
            features: [],
          },
        });

        map.addLayer({
          id: "crop-highlight-fill",
          type: "fill",
          source: "crop-highlight-source",
          paint: {
            "fill-color": "#10b981",
            "fill-opacity": 0.15,
          },
        });

        map.addLayer({
          id: "crop-highlight-line",
          type: "line",
          source: "crop-highlight-source",
          paint: {
            "line-color": "#34d399",
            "line-width": 2.5,
            "line-dasharray": [2, 1],
          },
        });
      }

      if (onMapReady) onMapReady(map);
      setMapReady(true);
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
      setMapReady(false);
    };
  }, []);

  // Update highlight box when selectedResult changes
  useEffect(() => {
    const map = mapInstance.current;
    if (!map || !map.isStyleLoaded()) return;

    const highlightSource = map.getSource("crop-highlight-source");
    if (!highlightSource) return;

    if (selectedResult && selectedResult.bounds && selectedResult.bounds.length === 4) {
      const [minx, miny, maxx, maxy] = selectedResult.bounds;
      const polygonGeoJSON = {
        type: "FeatureCollection",
        features: [
          {
            type: "Feature",
            geometry: {
              type: "Polygon",
              coordinates: [
                [
                  [minx, miny],
                  [maxx, miny],
                  [maxx, maxy],
                  [minx, maxy],
                  [minx, miny],
                ],
              ],
            },
            properties: {
              tile_id: selectedResult.tile_id,
              score: selectedResult.score,
            },
          },
        ],
      };
      highlightSource.setData(polygonGeoJSON);

      // Fly map to crop bounds
      map.fitBounds(
        [
          [minx, miny],
          [maxx, maxy],
        ],
        {
          padding: 80,
          duration: 1200,
          maxZoom: 17,
        }
      );
    } else {
      highlightSource.setData({
        type: "FeatureCollection",
        features: [],
      });
    }
  }, [selectedResult]);

  // Update raster layer when currentScene changes
  useEffect(() => {
    const map = mapInstance.current;
    if (!map || !mapReady || !currentScene || (!currentScene.cog_path && !currentScene.cog_url)) return;

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

        // Add raster layer beneath any vector highlight overlays
        const beforeLayerId = map.getLayer("crop-highlight-fill") ? "crop-highlight-fill" : undefined;

        map.addLayer(
          {
            id: layerId,
            type: "raster",
            source: sourceId,
            paint: {
              "raster-resampling": "linear",
              "raster-fade-duration": 200,
            },
          },
          beforeLayerId
        );

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
  }, [currentScene, mapReady]);

  return (
    <div className="relative w-full h-full flex-1 overflow-hidden bg-[#18181b]">
      <div ref={mapContainer} className="w-full h-full" />
    </div>
  );
});

export default MapView;
