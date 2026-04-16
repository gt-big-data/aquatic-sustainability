"""One-shot: reverse-geocode the 133 coral-cluster centroids and write centroid_names.json.

Run from the backend/ directory:
    python scripts/build_coral_centroid_names.py

Requires GOOGLE_MAPS_API_KEY in the environment (loaded via app.config.Config).
Idempotent — skips cluster_ids already present in the JSON.
"""

import csv
import json
import os
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from app.config import Config

CSVS = [
    os.path.join("data", "coral_bleaching", "gbr_predictions_temporal_lstm.csv"),
    os.path.join("data", "coral_bleaching", "gbr_predictions_temporal_xgboost.csv"),
]
NAMES_JSON = os.path.join("data", "coral_bleaching", "centroid_names.json")
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def extract_centroids() -> dict:
    centroids: dict = {}
    for path in CSVS:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                cid = str(int(row["cluster_id"]))
                if cid not in centroids:
                    centroids[cid] = (float(row["centroid_lat"]), float(row["centroid_lon"]))
    return dict(sorted(centroids.items(), key=lambda kv: int(kv[0])))


def load_existing() -> dict:
    try:
        with open(NAMES_JSON) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save(names: dict) -> None:
    os.makedirs(os.path.dirname(NAMES_JSON), exist_ok=True)
    ordered = {k: names[k] for k in sorted(names, key=int)}
    with open(NAMES_JSON, "w") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)
        f.write("\n")


def geocode(lat: float, lon: float, api_key: str) -> str | None:
    resp = requests.get(
        GEOCODE_URL,
        params={"latlng": f"{lat},{lon}", "key": api_key},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "OK":
        print(f"  ! non-OK status: {data.get('status')} ({data.get('error_message', '')})")
        return None
    results = data.get("results") or []
    if not results:
        return None
    return results[0].get("formatted_address")


def main() -> int:
    api_key = Config.GOOGLE_MAPS_API_KEY
    if not api_key:
        print("GOOGLE_MAPS_API_KEY is not set in the environment / .env", file=sys.stderr)
        return 1

    centroids = extract_centroids()
    names = load_existing()
    missing = [cid for cid in centroids if cid not in names]
    print(f"{len(centroids)} centroids total, {len(missing)} to geocode")

    for i, cid in enumerate(missing, 1):
        lat, lon = centroids[cid]
        try:
            name = geocode(lat, lon, api_key)
        except requests.RequestException as exc:
            print(f"  [{i}/{len(missing)}] cluster {cid}: request failed: {exc}")
            continue

        if name:
            names[cid] = name
            print(f"  [{i}/{len(missing)}] cluster {cid} ({lat:.3f}, {lon:.3f}) -> {name}")
        else:
            names[cid] = f"{lat:.3f}, {lon:.3f}"
            print(f"  [{i}/{len(missing)}] cluster {cid} ({lat:.3f}, {lon:.3f}) -> (no result, using coords)")

        # Persist incrementally so a crash/rate-limit doesn't lose progress.
        save(names)
        time.sleep(0.05)

    print(f"wrote {NAMES_JSON} with {len(names)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
