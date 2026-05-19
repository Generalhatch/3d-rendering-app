/**
 * Convert GeoJSON line features from the /plan endpoint into a Float32Array
 * of 3D line segment positions for THREE.LineSegments.
 */

interface GeoJSONFeature {
  type: string;
  geometry: {
    type: string;
    coordinates: number[][];
  };
}

interface GeoJSONCollection {
  type: string;
  features: GeoJSONFeature[];
}

export function planGeoJSONToSegments(geojson: GeoJSONCollection): Float32Array {
  const verts: number[] = [];

  for (const feature of geojson.features) {
    const geom = feature.geometry;
    if (geom.type === 'LineString') {
      const coords = geom.coordinates;
      for (let i = 0; i < coords.length - 1; i++) {
        const a = coords[i];
        const b = coords[i + 1];
        // Z = 0 for plan lines (they live in the XY plane)
        verts.push(a[0], a[1], 0, b[0], b[1], 0);
      }
    }
  }

  return new Float32Array(verts);
}
