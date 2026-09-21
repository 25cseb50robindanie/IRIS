import React, { useEffect, useRef, useState, useImperativeHandle, forwardRef } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

// Remove the scene layer(s) and their tile sources, leaving the highlight overlay in place
function removeCogLayers(map) {
  (map.getStyle().layers || []).forEach((l) => {
    if (l.id.startsWith("cog-layer-") && map.getLayer(l.id)) map.removeLayer(l.id);
  });
  Object.keys(map.getStyle().sources || {}).forEach((id) => {
    if (id.startsWith("cog-source-") && map.getSource(id)) map.removeSource(id);
  });
}

const boxRing = ([minx, miny, maxx, maxy]) => [
  [minx, miny],
  [maxx, miny],
  [maxx, maxy],
  [minx, maxy],
  [minx, miny],
];

const WATCH_COLORS = { quiet: "#2563eb", alert: "#dc2626" }; // a pinned place, and one with a detection the analyst has not seen
const LONG_PRESS_MS = 700;

// A pin at the centre of each watched box and the box itself; `alert` marks the ones with an unseen detection
function watchCollection(locations, alertIds) {
  const features = [];
  for (const loc of locations) {
    const [minx, miny, maxx, maxy] = loc.bounds;
    const alert = alertIds.has(loc.id);
    features.push({
      type: "Feature",
      properties: { id: loc.id, alert },
      geometry: { type: "Polygon", coordinates: [boxRing(loc.bounds)] },
    });
    features.push({
      type: "Feature",
      properties: { id: loc.id, alert },
      geometry: { type: "Point", coordinates: [(minx + maxx) / 2, (miny + maxy) / 2] },
    });
  }
  return { type: "FeatureCollection", features };
}

const OUTLINE_LAYERS = ["change-outline-fill", "change-outline-line"];
const OUTLINE_COLORS = { normal: "#f59e0b", ablation: "#dc2626" }; // amber: the full pipeline; red: suppression off
const ATTRIBUTION_OPACITY = 0.45; // the heatmap is a hint over the imagery, not a replacement for it


// GeoJSON for the outlines: the traced shape where there is one, else the box
function outlineCollection(items) {
  return {
    type: "FeatureCollection",
    features: items.map((c) => ({
      type: "Feature",
      properties: { id: c.id },
      geometry: c.geometry || { type: "Polygon", coordinates: [boxRing(c.bounds)] },
    })),
  };
}

/**
 * The map. Besides the scene and the selected-tile box it draws, on request:
 *   changeOutlines  {items: [{id, bounds, geometry?}], kind: "normal" | "ablation", selectedId}  outlines of changes
 *   attribution     {url, coordinates}  the heatmap of where a query matched, over the selected tile
 *   watchlist       the watched locations [{id, bounds}] drawn as pins, and alertWatchIds the ids with an unseen detection
 * onOutlineClick(id) is called for a click on an outline (leave it out and outlines are not clickable);
 * onWatchClick(id) for a click on a pin; onContextMenu({lng, lat, x, y}) for a right-click or long press;
 * onBackgroundClick() for a click on anything else.
 */
const MapView = forwardRef(function MapView(
  {
    currentScene,
    selectedResult,
    onMouseMove,
    onMapReady,
    changeOutlines = null,
    onOutlineClick = null,
    attribution = null,
    onBackgroundClick = null,
    watchlist = [],
    alertWatchIds = null,
    onWatchClick = null,
    onContextMenu = null,
  },
  ref
) {
  const mapContainer = useRef(null);
  const mapInstance = useRef(null);
  const activeBounds = useRef(null);
  // When the scene changes *because* the analyst picked a search result in it, the result's own fly-to must win
  const selectedResultRef = useRef(null);
  selectedResultRef.current = selectedResult;
  // Handlers and state the map's own event listeners (registered once) must always see the latest of
  const outlineClickRef = useRef(null);
  outlineClickRef.current = onOutlineClick;
  const backgroundClickRef = useRef(null);
  backgroundClickRef.current = onBackgroundClick;
  const outlinesActive = useRef(false);
  outlinesActive.current = Boolean(changeOutlines && changeOutlines.items.length);
  const attributionRef = useRef(null);
  attributionRef.current = attribution;
  const watchClickRef = useRef(null);
  watchClickRef.current = onWatchClick;
  const contextMenuRef = useRef(null);
  contextMenuRef.current = onContextMenu;
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

      // Outlines of detected changes; the selected one is drawn heavier
      map.addSource("change-outline-source", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer({
        id: "change-outline-fill",
        type: "fill",
        source: "change-outline-source",
        paint: { "fill-color": OUTLINE_COLORS.normal, "fill-opacity": 0.1 },
      });
      map.addLayer({
        id: "change-outline-line",
        type: "line",
        source: "change-outline-source",
        paint: { "line-color": OUTLINE_COLORS.normal, "line-width": 1.5 },
      });
      map.addLayer({
        id: "change-outline-selected",
        type: "line",
        source: "change-outline-source",
        filter: ["==", ["get", "id"], -1],
        paint: { "line-color": "#ffffff", "line-width": 3 },
      });

      // Watched locations: the box, and a pin at its centre (red while a detection there has not been seen)
      map.addSource("watch-source", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer({
        id: "watch-box",
        type: "line",
        source: "watch-source",
        filter: ["==", ["geometry-type"], "Polygon"],
        paint: {
          "line-color": ["case", ["get", "alert"], WATCH_COLORS.alert, WATCH_COLORS.quiet],
          "line-width": 1.5,
          "line-dasharray": [3, 2],
        },
      });
      map.addLayer({
        id: "watch-pin-halo",
        type: "circle",
        source: "watch-source",
        filter: ["==", ["geometry-type"], "Point"],
        paint: { "circle-radius": 13, "circle-color": ["case", ["get", "alert"], WATCH_COLORS.alert, WATCH_COLORS.quiet], "circle-opacity": 0.2 },
      });
      map.addLayer({
        id: "watch-pin",
        type: "circle",
        source: "watch-source",
        filter: ["==", ["geometry-type"], "Point"],
        paint: {
          "circle-radius": 6,
          "circle-color": ["case", ["get", "alert"], WATCH_COLORS.alert, WATCH_COLORS.quiet],
          "circle-stroke-color": "#ffffff",
          "circle-stroke-width": 2,
        },
      });

      // The browser's own menu has nothing to offer on a map
      const canvas = map.getCanvasContainer();
      canvas.addEventListener("contextmenu", (ev) => ev.preventDefault());
      const at = (point) => {
        const ll = map.unproject(point);
        return { lng: ll.lng, lat: ll.lat, x: point.x, y: point.y };
      };
      map.on("contextmenu", (e) => contextMenuRef.current && contextMenuRef.current(at(e.point)));
      // Long press does the same on a touch screen
      let press = null;
      const cancel = () => {
        clearTimeout(press);
        press = null;
      };
      canvas.addEventListener("touchstart", (ev) => {
        if (ev.touches.length !== 1) return cancel();
        const rect = canvas.getBoundingClientRect();
        const point = { x: ev.touches[0].clientX - rect.left, y: ev.touches[0].clientY - rect.top };
        press = setTimeout(() => contextMenuRef.current && contextMenuRef.current(at(point)), LONG_PRESS_MS);
      });
      canvas.addEventListener("touchmove", cancel);
      canvas.addEventListener("touchend", cancel);
      canvas.addEventListener("touchcancel", cancel);

      if (onMapReady) onMapReady(map);
      setMapReady(true);
    });

    map.on("click", (e) => {
      if (map.getLayer("watch-pin")) {
        const pins = map.queryRenderedFeatures(e.point, { layers: ["watch-pin", "watch-pin-halo"] });
        if (pins.length) {
          if (watchClickRef.current) watchClickRef.current(pins[0].properties.id);
          return;
        }
      }
      if (map.getLayer("change-outline-fill")) {
        const hits = map.queryRenderedFeatures(e.point, { layers: OUTLINE_LAYERS });
        if (hits.length) {
          if (outlineClickRef.current) outlineClickRef.current(hits[0].properties.id);
          return; // a click on an outline is not a click away
        }
      }
      if (backgroundClickRef.current) backgroundClickRef.current();
    });

    map.on("mousemove", (e) => {
      if (outlinesActive.current && map.getLayer("change-outline-fill")) {
        const over = outlineClickRef.current && map.queryRenderedFeatures(e.point, { layers: OUTLINE_LAYERS }).length > 0;
        map.getCanvas().style.cursor = over ? "pointer" : "";
      }
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

  // Outlines of the changes being looked at: amber for the full pipeline, red with the suppression off
  useEffect(() => {
    const map = mapInstance.current;
    if (!map || !mapReady) return;
    const source = map.getSource("change-outline-source");
    if (!source) return;
    const kind = changeOutlines?.kind === "ablation" ? "ablation" : "normal";
    source.setData(outlineCollection(changeOutlines?.items || []));
    for (const id of OUTLINE_LAYERS) {
      map.setPaintProperty(id, id.endsWith("fill") ? "fill-color" : "line-color", OUTLINE_COLORS[kind]);
    }
    map.setFilter("change-outline-selected", ["==", ["get", "id"], changeOutlines?.selectedId ?? -1]);
  }, [changeOutlines, mapReady]);

  // The watched locations
  const watchSignature = JSON.stringify([watchlist.map((w) => [w.id, w.bounds]), alertWatchIds ? [...alertWatchIds].sort() : []]);
  useEffect(() => {
    const map = mapInstance.current;
    if (!map || !mapReady) return;
    const source = map.getSource("watch-source");
    if (source) source.setData(watchCollection(watchlist, alertWatchIds || new Set()));
  }, [watchSignature, mapReady]); // eslint-disable-line react-hooks/exhaustive-deps

  // Attribution heatmap over the selected tile. It sits above the scene and below the outlines.
  useEffect(() => {
    const map = mapInstance.current;
    if (!map || !mapReady) return undefined;
    const clear = () => {
      if (map.getLayer("attribution-layer")) map.removeLayer("attribution-layer");
      if (map.getSource("attribution-source")) map.removeSource("attribution-source");
    };
    clear();
    if (!attribution) return undefined;
    map.addSource("attribution-source", { type: "image", url: attribution.url, coordinates: attribution.coordinates });
    map.addLayer(
      {
        id: "attribution-layer",
        type: "raster",
        source: "attribution-source",
        paint: { "raster-opacity": ATTRIBUTION_OPACITY, "raster-fade-duration": 300, "raster-resampling": "linear" },
      },
      "crop-highlight-fill"
    );
    return clear;
  }, [attribution, mapReady]);

  // Update raster layer when currentScene changes
  useEffect(() => {
    const map = mapInstance.current;
    if (!map || !mapReady) return;
    if (!currentScene || (!currentScene.cog_path && !currentScene.cog_url)) {
      // No scene (e.g. the last one was deleted): show the empty map, not the previous scene's tiles
      removeCogLayers(map);
      activeBounds.current = null;
      return;
    }

    const targetUrl = currentScene.cog_url || currentScene.cog_path;
    const sourceId = `cog-source-${currentScene.scene_id}`;
    const layerId = `cog-layer-${currentScene.scene_id}`;

    removeCogLayers(map);

    async function loadCog() {
      const skipFit = selectedResultRef.current?.scene_id === currentScene.scene_id;
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
        if (map.getLayer("attribution-layer")) map.moveLayer("attribution-layer", beforeLayerId);

        // Use bounds directly from tilejson or bounds_wgs84
        if (currentScene.bounds_wgs84 && currentScene.bounds_wgs84.length === 4) {
          const fitTarget = [
            [currentScene.bounds_wgs84[0], currentScene.bounds_wgs84[1]],
            [currentScene.bounds_wgs84[2], currentScene.bounds_wgs84[3]],
          ];
          activeBounds.current = fitTarget;
          if (!skipFit) {
            map.fitBounds(fitTarget, {
              padding: 40,
              duration: 1000,
              maxZoom: 16,
            });
          }
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
                if (!skipFit) {
                  map.fitBounds(fitTarget, {
                    padding: 40,
                    duration: 1000,
                    maxZoom: 16,
                  });
                }
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
