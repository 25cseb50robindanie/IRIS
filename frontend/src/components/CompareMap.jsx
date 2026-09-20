import React, { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

const TILE_SERVER = "http://localhost:8000";

const emptyBox = { type: "FeatureCollection", features: [] };

const boxFeature = ([minx, miny, maxx, maxy]) => ({
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      properties: {},
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
    },
  ],
});

/**
 * One side of the before/after comparison: a single scene served by titiler, with the change candidate's
 * bounding box drawn on top. The parent receives the map through onReady so it can keep both sides in sync.
 */
export default function CompareMap({ cogUrl, box, focusBounds, onReady }) {
  const container = useRef(null);
  const mapRef = useRef(null);

  useEffect(() => {
    if (!container.current) return undefined;

    const map = new maplibregl.Map({
      container: container.current,
      style: {
        version: 8,
        sources: {},
        layers: [{ id: "background", type: "background", paint: { "background-color": "#18181b" } }],
      },
      bounds: focusBounds
        ? [
            [focusBounds[0], focusBounds[1]],
            [focusBounds[2], focusBounds[3]],
          ]
        : undefined,
      fitBoundsOptions: { padding: 60, maxZoom: 16 },
      attributionControl: false,
    });
    mapRef.current = map;

    map.on("load", () => {
      map.addSource("scene", {
        type: "raster",
        url: `${TILE_SERVER}/tiles/cog/tilejson.json?url=${encodeURIComponent(cogUrl)}`,
        tileSize: 256,
      });
      map.addLayer({ id: "scene", type: "raster", source: "scene", paint: { "raster-fade-duration": 150 } });

      map.addSource("candidate", { type: "geojson", data: box ? boxFeature(box) : emptyBox });
      map.addLayer({
        id: "candidate-fill",
        type: "fill",
        source: "candidate",
        paint: { "fill-color": "#f59e0b", "fill-opacity": 0.12 },
      });
      map.addLayer({
        id: "candidate-line",
        type: "line",
        source: "candidate",
        paint: { "line-color": "#f59e0b", "line-width": 2 },
      });

      if (onReady) onReady(map);
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
    // The panes are rebuilt when the selected candidate changes (the parent re-keys this component)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cogUrl]);

  return <div ref={container} className="w-full h-full" />;
}
