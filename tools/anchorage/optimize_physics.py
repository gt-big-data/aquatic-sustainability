"""Compare five fixed physics recipes on validation, then score the frozen winner.

The 2025/2026 test split has been reported in earlier project work. This bounded
run does not make it a pristine holdout: it prohibits test scoring while selecting
the training objective and records the selection before the one final test score.
"""

import argparse
import copy
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from forecast import CACHE, SOURCE_BLOB, SOURCE_URL, atomic_write, pinned_dataset, read_archive, source_hash
from physics import ORIGINAL_CONFIG, evaluate_model, train_model


CANDIDATES = [
    ("original-mse", {}),
    ("mae", {"loss": "mae"}),
    ("huber-025", {"loss": "huber", "huber_beta": 0.25}),
    ("huber-050", {"loss": "huber", "huber_beta": 0.5}),
    ("mae-l2-001", {"loss": "mae", "weight_decay": 0.001}),
]


def write_json(path, value):
    atomic_write(path, lambda temporary: temporary.write_text(json.dumps(value, indent=2) + "\n"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=CACHE / "physics.joblib")
    parser.add_argument("--report-dir", type=Path, default=CACHE / "physics_optimization")
    args = parser.parse_args()
    started = time.perf_counter()
    dataset = pinned_dataset(args.dataset)
    archive = read_archive(dataset)
    with np.load(dataset, allow_pickle=False) as source:
        windows, split = source["windows"], source["split"]
    args.report_dir.mkdir(parents=True, exist_ok=True)
    records = []
    winner = None
    winner_name = None
    for name, overrides in CANDIDATES:
        config = copy.deepcopy(ORIGINAL_CONFIG)
        config.update(overrides)
        print(f"Training validation-only candidate: {name}", flush=True)
        model = train_model(archive, dataset, windows, split, config=config, evaluate_test=False)
        assert set(model["evaluation"]) == {"validation"}
        record = {
            "name": name, "config": config, "validation": model["evaluation"]["validation"],
            "fit_seconds": model["fit_seconds"],
            "selected_epochs": [member["best_epoch"] for member in model["members"]],
            "kappa_m2_day": [member["kappa_m2_day"] for member in model["members"]],
            "training_history": model["training_history"],
        }
        records.append(record)
        atomic_write(args.report_dir / f"{name}.joblib", lambda path: joblib.dump(model, path, compress=3))
        write_json(args.report_dir / "validation_candidates.json", records)
        if winner is None or record["validation"]["mae_ft"] < winner["evaluation"]["validation"]["mae_ft"]:
            winner, winner_name = model, name
        print(f"Candidate {name}: validation MAE {record['validation']['mae_ft']:.9f} ft", flush=True)

    selection = {
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "criterion": "Minimum two-seed ensemble mean per-window validation MAE in feet; candidate order breaks exact ties.",
        "winner": winner_name,
        "chosen_config": next(item["config"] for item in records if item["name"] == winner_name),
        "validation_mae_ft": winner["evaluation"]["validation"]["mae_ft"],
        "test_evaluations_during_selection": 0,
        "historical_test_exposure": "The 2025/2026 test results were previously inspected; this is validation-only retuning, not a newly untouched holdout.",
    }
    write_json(args.report_dir / "selection_before_test.json", selection)
    write_json(args.report_dir / "selected_config.json", selection["chosen_config"])
    # This is the sole test-scoring call in the search, after selection is durable.
    winner["evaluation"]["test"] = evaluate_model(winner, archive, dataset, windows[split == "test"])
    winner["optimization_selection"] = selection
    atomic_write(args.output, lambda path: joblib.dump(winner, path, compress=3))
    provenance = {
        "dataset_path": str(dataset.resolve()), "dataset_git_blob": source_hash(dataset),
        "expected_dataset_git_blob": SOURCE_BLOB, "dataset_source_url": SOURCE_URL,
        "survey_dates": [str(archive.dates.min()), str(archive.dates.max())],
        "split_counts": {label: int((split == label).sum()) for label in ("train", "val", "test")},
        "target_date_ranges": {
            label: [str(archive.dates[windows[split == label, -1]].min()), str(archive.dates[windows[split == label, -1]].max())]
            for label in ("train", "val", "test")
        },
        "core_cells": int(winner["core"].sum()), "training_rows": winner["training_rows"],
        "python": platform.python_version(), "numpy": np.__version__,
        "sklearn": winner["sklearn"], "torch": str(winner["torch"]),
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("physics.py", "train_physics.py", "optimize_physics.py")
        },
    }
    summary = {
        "source_provenance": provenance, "selection": selection, "candidates": records,
        "selected_model_evaluation": winner["evaluation"], "artifact": str(args.output.resolve()),
        "test_evaluations_total": 1, "total_seconds": time.perf_counter() - started,
        "constant_footprint": {"seeds": ORIGINAL_CONFIG["seeds"], "epochs": ORIGINAL_CONFIG["epochs"],
                               "network": [84, 32, 32, 1], "sample_cells_per_training_window": 400},
    }
    write_json(args.report_dir / "summary.json", summary)
    print(json.dumps({"winner": winner_name, "evaluation": winner["evaluation"], "report": str(args.report_dir / 'summary.json')}, indent=2), flush=True)


if __name__ == "__main__":
    main()
