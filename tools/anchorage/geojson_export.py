"""The shared geographic output contract for Anchorage model adapters."""

import json
from pathlib import Path

import numpy as np
from pyproj import Transformer

from forecast import FEET_TO_METRES


REFERENCE_CELLS = Path(__file__).with_name("reference_cells.json")


def reference_footprint(reference_path, grid):
    """Extract grid locations from a supplied GeoJSON without reading its predictions."""
    collection = json.loads(Path(reference_path).read_text())
    to_grid = Transformer.from_crs("EPSG:4326", grid["crs"], always_xy=True)
    cells = set()
    spacing = grid["cell_metres"]
    for feature in collection["features"]:
        geometry = feature["geometry"]
        if geometry["type"] != "Polygon" or len(geometry["coordinates"]) != 1:
            raise ValueError("The reference footprint must contain grid-cell polygons.")
        ring = geometry["coordinates"][0]
        if len(ring) != 5 or ring[0] != ring[-1]:
            raise ValueError("Each reference cell must have four corners and a closed ring.")
        x, y = to_grid.transform(*np.asarray(ring[:4]).T)
        southwest_x, southwest_y = min(x), min(y)
        col = round(southwest_x / spacing) - grid["col0"]
        row = round(southwest_y / spacing) - grid["row0"]
        expected_x = (grid["col0"] + col) * spacing
        expected_y = (grid["row0"] + row) * spacing
        if not (0 <= row < grid["rows"] and 0 <= col < grid["cols"]):
            raise ValueError("A reference cell lies outside the model grid.")
        if max(abs(southwest_x - expected_x), abs(southwest_y - expected_y),
               abs(max(x) - min(x) - spacing), abs(max(y) - min(y) - spacing)) > 0.02:
            raise ValueError("The reference polygons do not align with the model grid.")
        cells.add((row, col))
    if len(cells) != len(collection["features"]):
        raise ValueError("The reference footprint contains duplicate cells.")
    return sorted(cells)


def eligible_cells(model, archive, history, predicted_change):
    rows, cols = np.where(model["core"])
    predicted_change = np.asarray(predicted_change)
    if predicted_change.shape != rows.shape:
        raise ValueError("The adapter must predict one value per declared core cell.")
    origin = archive.maps[history[-1]][model["core"]]
    valid = np.isfinite(predicted_change) & np.isfinite(origin)
    return set(zip(rows[valid].tolist(), cols[valid].tolist()))


def feature_collection(model, archive, history, target_date, predicted_change, footprint):
    rows, cols = np.where(model["core"])
    cell_index = {(int(row), int(col)): index for index, (row, col) in enumerate(zip(rows, cols))}
    origin = archive.maps[history[-1]][model["core"]]
    to_lonlat = Transformer.from_crs(archive.grid["crs"], "EPSG:4326", always_xy=True)
    spacing = archive.grid["cell_metres"]
    features = []
    for row, col in sorted(footprint):
        index = cell_index[(row, col)]
        change = float(predicted_change[index])
        last = float(origin[index])
        if not np.isfinite([change, last]).all():
            raise ValueError("The shared footprint contains an unavailable prediction.")
        x = (archive.grid["col0"] + col) * spacing
        y = (archive.grid["row0"] + row) * spacing
        corners = [(x, y), (x + spacing, y), (x + spacing, y + spacing), (x, y + spacing), (x, y)]
        ring = [list(to_lonlat.transform(easting, northing)) for easting, northing in corners]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "depth_ft": round((last + change) / FEET_TO_METRES, 2),
                "change_ft": round(change / FEET_TO_METRES, 2),
            },
        })
    if not features:
        raise ValueError("The models have no common predicted footprint.")
    return {
        "type": "FeatureCollection",
        "properties": {
            "from_survey": str(archive.dates[history[-1]]),
            "predicted_for": str(target_date),
        },
        "features": features,
    }
