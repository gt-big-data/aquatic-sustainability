"""Train a bounded XGBoost search; freeze validation selection before test scoring."""

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np

from forecast import (CACHE, FEET_TO_METRES, atomic_write, evaluate, fit_model,
                      pinned_dataset, predict_change, read_archive)

OUTPUT = CACHE / "xgboost_optimization"
CANDIDATES = [
    ("incumbent", {}),
    ("sample1200", {"cells_per_window": 1200}),
    ("all_cells", {"cells_per_window": 10000}),
    ("details1200", {"cells_per_window": 1200, "feature_set": "details"}),
    ("drivers1200", {"cells_per_window": 1200, "feature_set": "drivers"}),
    ("details_rate1200", {"cells_per_window": 1200, "feature_set": "details", "rate_target": True}),
    ("details_rate_all", {"cells_per_window": 10000, "feature_set": "details", "rate_target": True}),
    ("drivers_rate_all", {"cells_per_window": 10000, "feature_set": "drivers", "rate_target": True}),
    ("details_shallow", {"cells_per_window": 10000, "feature_set": "details", "rate_target": True,
                         "parameters": {"max_depth": 5}}),
    ("details_deep", {"cells_per_window": 10000, "feature_set": "details", "rate_target": True,
                      "parameters": {"max_depth": 9}}),
    ("details_mae", {"cells_per_window": 10000, "feature_set": "details", "rate_target": True,
                     "parameters": {"objective": "reg:absoluteerror"}}),
    ("details_huber05", {"cells_per_window": 10000, "feature_set": "details", "rate_target": True,
                         "parameters": {"huber_slope": 0.5}}),
]


def write_json(path, value):
    atomic_write(path, lambda temporary: temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    dataset = pinned_dataset(args.dataset)
    archive = read_archive(dataset)
    with np.load(dataset, allow_pickle=False) as source:
        windows, split = source["windows"], source["split"]
    validation = windows[split == "val"]
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for name, config in CANDIDATES:
        started = time.perf_counter()
        model = fit_model(archive, windows, split, config)
        scores = evaluate(model, archive, validation)
        record = {"id": name, "config": config, "validation": scores,
                  "training_rows": model["training_rows"], "features": len(model["input_features"]),
                  "boosting_rounds": int(model["estimator"].best_iteration + 1),
                  "fit_validation_seconds": time.perf_counter() - started}
        results.append(record)
        atomic_write(args.output / f"{name}.joblib", lambda path: joblib.dump(model, path, compress=3))
        write_json(args.output / "validation.json", results)
        print(json.dumps(record), flush=True)
        del model
        gc.collect()

    winner = min(results, key=lambda row: row["validation"]["mae_ft"])
    selected = joblib.load(args.output / f"{winner['id']}.joblib")
    # A second fixed seed checks whether averaging improves validation.
    config43 = copy.deepcopy(winner["config"])
    config43.setdefault("parameters", {})["random_state"] = 43
    second = fit_model(archive, windows, split, config43)
    ensemble = dict(selected)
    ensemble["estimators"] = [selected["estimator"], second["estimator"]]
    ensemble_score = evaluate(ensemble, archive, validation)
    use_ensemble = ensemble_score["mae_ft"] < winner["validation"]["mae_ft"]
    final = ensemble if use_ensemble else selected
    final_config = {"config": winner["config"], "seeds": [42, 43] if use_ensemble else [42]}
    selection = {"selection_metric": "Mean per-window validation MAE in feet",
                 "test_status": "2025-2026 was inspected previously; no test labels used in this search or selection.",
                 "winner": winner["id"], "ensemble_validation": ensemble_score,
                 "use_ensemble": use_ensemble, "recipe": final_config, "candidates": results}
    write_json(args.output / "selection_before_test.json", selection)
    final["selection"] = {"winner": winner["id"], "seeds": final_config["seeds"],
                          "criterion": "validation_mae_ft", "test_previously_inspected": True}
    final["evaluation"] = {"validation": ensemble_score if use_ensemble else winner["validation"],
                           "test": evaluate(final, archive, windows[split == "test"])}
    incumbent = joblib.load(args.output / "incumbent.joblib")
    selection["final_evaluation"] = final["evaluation"]
    selection["incumbent_evaluation"] = {"validation": results[0]["validation"],
                                        "test": evaluate(incumbent, archive, windows[split == "test"])}
    write_json(args.output / "summary.json", selection)
    write_json(Path(__file__).with_name("release_config.json"), final_config)
    atomic_write(CACHE / "xgboost.joblib", lambda path: joblib.dump(final, path, compress=3))
    print(json.dumps({"winner": winner["id"], "ensemble": use_ensemble,
                      "evaluation": final["evaluation"]}), flush=True)


if __name__ == "__main__":
    main()
