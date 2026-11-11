"""Publish model-specific predictions through the common GeoJSON contract."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from forecast import (CACHE, HERE, LEAD_DAYS, atomic_write, pinned_dataset,
                      predict_change, read_archive)
from geojson_export import REFERENCE_CELLS, eligible_cells, feature_collection


STATIC = HERE.parents[1] / "backend/static"
DEFAULT_MANIFEST = STATIC / "data/anchorage/manifest.json"


@dataclass(frozen=True)
class Adapter:
    id: str
    label: str
    artifact: str

    def predict(self, model, archive, dataset_path, history):
        if self.id == "physics":
            from physics import predict_change as physics_prediction

            return physics_prediction(model, archive, dataset_path, history, LEAD_DAYS)
        return predict_change(model, archive, history, LEAD_DAYS)


ADAPTERS = (
    Adapter("xgboost", "XGBoost + neighborhoods", "xgboost.joblib"),
    Adapter("physics", "Physics-informed diffusion + neural source", "physics.joblib"),
)


def write_json(path, value):
    atomic_write(path, lambda temporary: temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"))


def history_at_dates(archive, history_dates):
    indexes = np.searchsorted(archive.dates, history_dates)
    if (indexes >= len(archive.dates)).any() or not np.array_equal(archive.dates[indexes], history_dates):
        raise ValueError("An adapter's input archive lacks a required historical survey date.")
    return indexes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--xgboost-input", type=Path, help="XGBoost-specific v2 input archive.")
    parser.add_argument("--physics-input", type=Path, help="Physics-specific v2 archive with daily drivers.")
    parser.add_argument("--artifacts", type=Path, default=CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--origin", help="Publish a 28-day projection from this origin instead of evaluated frames.")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive.")
    default_input = pinned_dataset(args.dataset)
    inputs = {"xgboost": args.xgboost_input or default_input, "physics": args.physics_input or default_input}
    reference = {tuple(cell) for cell in json.loads(REFERENCE_CELLS.read_text())}
    loaded = {}
    fingerprint = hashlib.sha256(REFERENCE_CELLS.read_bytes())
    for adapter in ADAPTERS:
        artifact_path = args.artifacts / adapter.artifact
        if not artifact_path.is_file():
            raise ValueError(f"Run train_{adapter.id}.py first to create {artifact_path}.")
        fingerprint.update(artifact_path.read_bytes())
        fingerprint.update(inputs[adapter.id].read_bytes())
        model = joblib.load(artifact_path)
        archive = read_archive(inputs[adapter.id])
        if model["grid"] != archive.grid:
            raise ValueError(f"Align {adapter.id} input maps to its trained grid.")
        loaded[adapter.id] = model, archive

    first_model, primary = loaded["xgboost"]
    if args.origin:
        origin = np.datetime64(args.origin, "D")
        found = np.flatnonzero(primary.dates == origin)
        if len(found) != 1 or int(found[0]) < 3 * primary.step:
            raise ValueError("Choose an available origin with four historical maps.")
        history = int(found[0]) - np.arange(3, -1, -1) * primary.step
        frames = [(history, origin + np.timedelta64(LEAD_DAYS, "D"), "forecast")]
    else:
        with np.load(inputs["xgboost"], allow_pickle=False) as source:
            windows = source["windows"][source["split"] == "test"][-args.frames:]
        frames = [(window[:-1], primary.dates[window[-1]], "hindcast") for window in windows]

    fingerprint.update(str([(history.tolist(), str(target)) for history, target, _ in frames]).encode())
    generation = fingerprint.hexdigest()[:16]
    manifest = {
        "schema_version": 1,
        "models": [{"id": adapter.id, "label": adapter.label,
                    "mae_ft": float(loaded[adapter.id][0]["evaluation"]["test"]["mae_ft"])} for adapter in ADAPTERS],
        "frames": [],
        "cell_metres": primary.grid["cell_metres"],
        "source_commit": first_model["source_commit"],
    }
    report = {
        "source_commit": first_model["source_commit"],
        "metric": "Mean per-survey MAE in feet over observed origin/target pairs in the training-defined core.",
        "splits": {"train": "2014-2022", "validation": "2023-2024", "test": "2025-2026"},
        "models": {},
    }
    for adapter in ADAPTERS:
        model, _ = loaded[adapter.id]
        details = {
            "training_script": f"train_{adapter.id}.py",
            "training_rows": model["training_rows"],
            "training_windows": model["training_windows"],
            "train_end": model["train_end"],
            "core_cells": int(model["core"].sum()),
            "features": model["input_features"],
            "evaluation": model["evaluation"],
        }
        if adapter.id == "physics":
            details["physics"] = model["physics"]
            details["members"] = [{key: member[key] for key in ("seed", "best_epoch", "kappa_m2_day")}
                                  for member in model["members"]]
        else:
            details["boosting_rounds"] = int(model["estimator"].best_iteration + 1)
            details["feature_importance_gain"] = dict(zip(model["input_features"],
                model["estimator"].feature_importances_.astype(float).tolist()))
        report["models"][adapter.id] = details
    for primary_history, target, kind in frames:
        history_dates = primary.dates[primary_history]
        origin = history_dates[-1]
        if int((target - origin).astype(int)) != LEAD_DAYS:
            raise ValueError("The overlay registry supports the evaluated 28-day horizon.")
        predictions = {}
        footprint = reference.copy()
        for adapter in ADAPTERS:
            model, archive = loaded[adapter.id]
            if str(origin) <= model["train_end"]:
                raise ValueError("Forecast origin must be after the training period.")
            month = target.astype("datetime64[M]").astype(int) % 12 + 1
            if month not in model["target_months"]:
                raise ValueError(f"Target {target} falls outside {adapter.id}'s evaluated season.")
            history = history_at_dates(archive, history_dates)
            if np.diff(archive.dates[history]).astype(int).max() > model["max_history_gap"]:
                raise ValueError("Historical survey intervals exceed the trained range.")
            delta = adapter.predict(model, archive, inputs[adapter.id], history)
            footprint &= eligible_cells(model, archive, history, delta)
            predictions[adapter.id] = history, delta

        frame = {"id": str(origin), "from_survey": str(origin), "predicted_for": str(target),
                 "kind": kind, "outputs": {}}
        for adapter in ADAPTERS:
            model, archive = loaded[adapter.id]
            history, delta = predictions[adapter.id]
            collection = feature_collection(model, archive, history, target, delta, footprint)
            output = args.manifest.parent / "predictions" / generation / f"{adapter.id}-{origin}.geojson"
            write_json(output, collection)
            frame["outputs"][adapter.id] = "/static/" + output.resolve().relative_to(STATIC.resolve()).as_posix()
        manifest["frames"].append(frame)
    # New generations use immutable asset URLs. The manifest switches after every file is ready.
    write_json(args.manifest.with_name("evaluation.json"), report)
    write_json(args.manifest, manifest)
    print(json.dumps({"manifest": str(args.manifest), "models": manifest["models"],
                      "frames": len(manifest["frames"]), "generation": generation}, indent=2))


if __name__ == "__main__":
    main()
