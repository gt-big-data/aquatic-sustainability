"""Loader for coral bleaching prediction CSVs.

Serves the factored JSON bundle consumed by /api/coral-bleaching.
"""

import csv
import json
import os
from datetime import datetime, timezone

_LSTM_CSV    = os.path.join("data", "coral_bleaching", "gbr_predictions_temporal_lstm.csv")
_XGBOOST_CSV = os.path.join("data", "coral_bleaching", "gbr_predictions_temporal_xgboost.csv")
_NAMES_JSON  = os.path.join("data", "coral_bleaching", "centroid_names.json")

_KELVIN_TO_C = 273.15

_METRIC_LABELS = {
    "risk_score":   {"unit": "",          "label": "Risk Score"},
    "tsa_dhw_last": {"unit": "\u00B0C-weeks", "label": "DHW (last)"},
    "filled_sst_c": {"unit": "\u00B0C",       "label": "SST"},
}

_bundle_cache = None
_bundle_mtime = None


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _sources_mtime() -> tuple:
    return (_mtime(_LSTM_CSV), _mtime(_XGBOOST_CSV), _mtime(_NAMES_JSON))


def _round(v, ndigits: int):
    if v is None:
        return None
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return None


def _read_model_csv(path: str):
    """Return (rows_by_cluster, weeks_sorted, centroids_by_cluster).

    rows_by_cluster: dict[cluster_id -> dict[date -> parsed_row]]
    weeks_sorted:    list of ISO dates in ascending order
    centroids_by_cluster: dict[cluster_id -> (lat, lon)]
    """
    rows_by_cluster: dict = {}
    weeks: set = set()
    centroids_by_cluster: dict = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            cluster_id = int(raw["cluster_id"])
            date = raw["date"]
            weeks.add(date)
            lat = float(raw["centroid_lat"])
            lon = float(raw["centroid_lon"])
            centroids_by_cluster.setdefault(cluster_id, (lat, lon))

            sst_k = float(raw["FilledSST_last"])
            parsed = {
                "predicted_class": int(raw["predicted_class"]),
                "risk_score":      float(raw["risk_score"]),
                "tsa_dhw_last":    float(raw["TSA_DHW_last"]),
                "tsa_dhw_max":     float(raw["TSA_DHW_max"]),
                "filled_sst_c":    sst_k - _KELVIN_TO_C,
                "prob_none":       float(raw["prob_none"]),
                "prob_moderate":   float(raw["prob_moderate"]),
                "prob_severe":     float(raw["prob_severe"]),
                "n_records":       int(float(raw["n_records"])),
            }
            rows_by_cluster.setdefault(cluster_id, {})[date] = parsed

    return rows_by_cluster, sorted(weeks), centroids_by_cluster


def _load_names() -> dict:
    try:
        with open(_NAMES_JSON) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    # normalize keys to str (JSON keys are always strings anyway)
    return {str(k): v for k, v in raw.items()}


def _build_bundle() -> dict:
    lstm_rows,    weeks_lstm,    centroids_lstm    = _read_model_csv(_LSTM_CSV)
    xgboost_rows, weeks_xgboost, centroids_xgboost = _read_model_csv(_XGBOOST_CSV)

    if weeks_lstm != weeks_xgboost:
        raise RuntimeError("coral bleaching CSVs have mismatched week grids")

    weeks = weeks_lstm
    cluster_ids = sorted(centroids_lstm.keys())

    names = _load_names()
    centroids = []
    for cid in cluster_ids:
        lat, lon = centroids_lstm[cid]
        name = names.get(str(cid)) or f"{lat:.3f}, {lon:.3f}"
        centroids.append({
            "cluster_id": cid,
            "lat": _round(lat, 5),
            "lon": _round(lon, 5),
            "name": name,
        })

    lats = [c["lat"] for c in centroids]
    lons = [c["lon"] for c in centroids]
    bbox = {
        "min_lat": min(lats), "max_lat": max(lats),
        "min_lon": min(lons), "max_lon": max(lons),
    }

    # Precision: probabilities & risk score → 3dp, DHW & SST → 2dp, n_records → int
    metric_precision = {
        "predicted_class": None,  # int
        "risk_score":      3,
        "tsa_dhw_last":    2,
        "tsa_dhw_max":     2,
        "filled_sst_c":    2,
        "prob_none":       3,
        "prob_moderate":   3,
        "prob_severe":     3,
        "n_records":       None,  # int
    }

    def build_model(rows_by_cluster):
        out = {k: [] for k in metric_precision}
        for cid in cluster_ids:
            per_date = rows_by_cluster.get(cid, {})
            series = {k: [] for k in metric_precision}
            for date in weeks:
                row = per_date.get(date)
                if row is None:
                    for k in metric_precision:
                        series[k].append(None)
                    continue
                for k, ndigits in metric_precision.items():
                    v = row[k]
                    series[k].append(v if ndigits is None else _round(v, ndigits))
            for k in metric_precision:
                out[k].append(series[k])
        return out

    models = {
        "lstm":    build_model(lstm_rows),
        "xgboost": build_model(xgboost_rows),
    }

    # Global min/max across both models for the three color-coded metrics.
    def global_extrema(key: str):
        vals = []
        for model in models.values():
            for series in model[key]:
                for v in series:
                    if v is not None:
                        vals.append(v)
        if not vals:
            return 0.0, 1.0
        return min(vals), max(vals)

    metrics = {}
    for key in ("risk_score", "tsa_dhw_last", "filled_sst_c"):
        lo, hi = global_extrema(key)
        ndigits = 3 if key == "risk_score" else 2
        metrics[key] = {
            "min":   _round(lo, ndigits),
            "max":   _round(hi, ndigits),
            "unit":  _METRIC_LABELS[key]["unit"],
            "label": _METRIC_LABELS[key]["label"],
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weeks": weeks,
        "bbox": bbox,
        "metrics": metrics,
        "severity_labels": ["None (0%)", "Moderate (1-50%)", "Severe (>50%)"],
        "centroids": centroids,
        "models": models,
    }


def get_bundle() -> dict:
    """Return the cached bundle, rebuilding if any source file changed."""
    global _bundle_cache, _bundle_mtime
    current = _sources_mtime()
    if _bundle_cache is None or current != _bundle_mtime:
        _bundle_cache = _build_bundle()
        _bundle_mtime = current
    return _bundle_cache
