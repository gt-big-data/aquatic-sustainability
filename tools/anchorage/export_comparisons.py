"""Export retrospective survey comparisons without changing forecast artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "backend/static"


def export(dataset, manifest_path, output):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(dataset, allow_pickle=False) as source:
        maps, dates = source["maps"], source["dates"].astype("datetime64[D]")
        row0, col0, cell = int(source["row0"]), int(source["col0"]), int(source["cell"])
    projection = Transformer.from_crs("EPSG:4326", "EPSG:26934", always_xy=True)
    frames = []
    for frame in manifest["frames"]:
        target = np.flatnonzero(dates == np.datetime64(frame["predicted_for"]))
        origin = np.flatnonzero(dates == np.datetime64(frame["from_survey"]))
        if len(target) != 1 or len(origin) != 1:
            raise ValueError(f"Exact origin and target surveys required: {frame['id']}")
        record = {"id": frame["id"], "from_survey": frame["from_survey"], "predicted_for": frame["predicted_for"], "outputs": {}}
        for model_id, url in frame["outputs"].items():
            if not url.startswith("/static/data/anchorage/"):
                raise ValueError("Forecast must be a local Anchorage asset")
            raw = (STATIC / url.removeprefix("/static/")).read_bytes()
            collection = json.loads(raw)
            if collection["properties"] != {"from_survey": frame["from_survey"], "predicted_for": frame["predicted_for"]}:
                raise ValueError("Forecast dates do not match manifest")
            observations = []
            for feature in collection["features"]:
                ring = np.asarray(feature["geometry"]["coordinates"][0][:4])
                x, y = projection.transform(ring[:, 0], ring[:, 1])
                col, row = round(float(x.min()) / cell) - col0, round(float(y.min()) / cell) - row0
                if not (0 <= row < maps.shape[1] and 0 <= col < maps.shape[2]):
                    raise ValueError("Polygon is outside the survey grid")
                if max(abs(float(x.min()) - (col0 + col) * cell), abs(float(y.min()) - (row0 + row) * cell)) > 0.02:
                    raise ValueError("Polygon does not align with survey grid")
                start, end = float(maps[origin[0], row, col]), float(maps[target[0], row, col])
                if np.isfinite(start) and np.isfinite(end):
                    observations.append([round(start, 5), round(end, 5)])
                else:
                    observations.append(None)
            record["outputs"][model_id] = {
                "forecast_url": url, "forecast_sha256": hashlib.sha256(raw).hexdigest(),
                "origin_target_depths_ft": observations,
            }
        frames.append(record)
    payload = {"schema_version": 1, "kind": "retrospective_observations",
               "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(), "frames": frames}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"Exported exact-survey comparisons for {len(frames)} frames to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "anchorage/data/timeseries_v2/timeseries.npz")
    parser.add_argument("--manifest", type=Path, default=STATIC / "data/anchorage/manifest.json")
    parser.add_argument("--output", type=Path, default=STATIC / "data/anchorage/comparisons.json")
    args = parser.parse_args()
    export(args.dataset, args.manifest, args.output)
