"""Build reproducible model comparisons and figures after validation selection."""

import argparse
import csv
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from forecast import CACHE, HERE, FEET_TO_METRES, pinned_dataset, predict_change, read_archive
from physics import predict_change as physics_prediction

OUTPUT = HERE.parents[1] / "backend/static/data/anchorage/optimization"
LABELS = {"incumbent": "Original XGBoost", "xgboost": "Final XGBoost",
          "physics_original": "Original physics", "physics": "Final physics", "persistence": "Persistence"}
COLORS = {"incumbent": "#8ca2b2", "xgboost": "#2166ac", "physics_original": "#d3ae87",
          "physics": "#bd6c29", "persistence": "#747474"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    dataset = pinned_dataset(args.dataset)
    archive = read_archive(dataset)
    with np.load(dataset, allow_pickle=False) as source:
        windows, split = source["windows"], source["split"]
    args.output.mkdir(parents=True, exist_ok=True)
    xgb_search = json.loads((CACHE / "xgboost_optimization/summary.json").read_text())
    physics_search = json.loads((CACHE / "physics_optimization/summary.json").read_text())
    models = {
        "incumbent": joblib.load(CACHE / "xgboost_optimization/incumbent.joblib"),
        "xgboost": joblib.load(CACHE / "xgboost.joblib"),
        "physics_original": joblib.load(CACHE / "physics_optimization/original-mse.joblib"),
        "physics": joblib.load(CACHE / "physics.joblib"),
    }
    core = models["xgboost"]["core"]
    assert all(np.array_equal(model["core"], core) for model in models.values())
    rows, errors, truths, predictions = [], {name: [] for name in LABELS}, [], {name: [] for name in LABELS}
    latest = {}
    for window in windows[split == "test"]:
        history, origin, target = window[:-1], window[-2], window[-1]
        truth = (archive.maps[target] - archive.maps[origin])[core] / FEET_TO_METRES
        measured = np.isfinite(truth)
        truths.append(truth[measured])
        for name in LABELS:
            if name == "persistence":
                predicted = np.zeros(int(core.sum()))
            elif name.startswith("physics"):
                predicted = physics_prediction(models[name], archive, dataset, history) / FEET_TO_METRES
            else:
                predicted = predict_change(models[name], archive, history) / FEET_TO_METRES
            assert np.isfinite(predicted[measured]).all()
            error = predicted[measured] - truth[measured]
            errors[name].append(error)
            predictions[name].append(predicted[measured])
            rows.append({"model": name, "from_survey": str(archive.dates[origin]),
                         "predicted_for": str(archive.dates[target]), "cells": int(measured.sum()),
                         "mae_ft": float(np.abs(error).mean()), "rmse_ft": float(np.sqrt(np.mean(error**2))),
                         "bias_ft": float(error.mean())})
            latest[name] = predicted
    write_csv(args.output / "per_window.csv", rows)
    truth = np.concatenate(truths)
    summaries, size_rows = {}, []
    for name in LABELS:
        error = np.concatenate(errors[name])
        predicted = np.concatenate(predictions[name])
        model_rows = [row for row in rows if row["model"] == name]
        actual_hot, predicted_hot = truth <= -1, predicted <= -1
        true_positive = int((actual_hot & predicted_hot).sum())
        summaries[name] = {
            "mae_ft": float(np.mean([row["mae_ft"] for row in model_rows])),
            "pooled_mae_ft": float(np.abs(error).mean()), "rmse_ft": float(np.sqrt(np.mean(error**2))),
            "bias_ft": float(error.mean()), "windows": len(model_rows), "scored_cells": len(error),
            "year_mae_ft": {str(year): float(np.mean([row["mae_ft"] for row in model_rows if row["predicted_for"].startswith(str(year))])) for year in (2025, 2026)},
            "hotspots": {"threshold_change_ft": -1, "actual_cells": int(actual_hot.sum()),
                         "predicted_cells": int(predicted_hot.sum()), "true_positive_cells": true_positive,
                         "precision": true_positive / int(predicted_hot.sum()) if predicted_hot.any() else None,
                         "recall": true_positive / int(actual_hot.sum()) if actual_hot.any() else None},
        }
        for label, low, high in (("<1 ft", 0, 1), ("1–2 ft", 1, 2), (">=2 ft", 2, np.inf)):
            mask = (np.abs(truth) >= low) & (np.abs(truth) < high)
            size_rows.append({"model": name, "actual_change_size": label, "cells": int(mask.sum()),
                              "mae_ft": float(np.abs(error[mask]).mean())})
    write_csv(args.output / "change_size.csv", size_rows)
    dates = np.asarray([row["predicted_for"] for row in rows if row["model"] == "xgboost"])
    months = np.unique([value[:7] for value in dates])
    paired = np.asarray([row["mae_ft"] for row in rows if row["model"] == "xgboost"]) - np.asarray([row["mae_ft"] for row in rows if row["model"] == "incumbent"])
    month_indexes = [np.flatnonzero(np.char.startswith(dates, month)) for month in months]
    rng = np.random.default_rng(42)
    bootstrap = [paired[np.concatenate([month_indexes[index] for index in rng.integers(len(months), size=len(months))])].mean() for _ in range(2000)]
    comparisons = {"final_minus_original_xgb_mae_ft": float(paired.mean()),
                   "calendar_month_block_bootstrap_95_ft": np.quantile(bootstrap, [.025, .975]).tolist(),
                   "month_blocks": len(months), "scope": "Descriptive interval on the previously inspected 2025-2026 backtest."}
    report = {
        "dataset_git_blob": "7b603395ef4992b201bc3dc00ef8c3ca1a22c521",
        "reviewed_upstream_commit": "016f0469a7a34756b6dc6d35b757511a7a630fd0",
        "metric": "Primary MAE averages measured cells within each survey, then averages surveys equally.",
        "selection": "All new choices use 2023-2024 validation. The 2025-2026 backtest was inspected in prior work and is reused for final comparison.",
        "split_counts": {label: int((split == label).sum()) for label in ("train", "val", "test")},
        "test": summaries, "paired_comparison": comparisons,
        "xgboost": xgb_search,
        "physics": {"selection": physics_search["selection"],
                    "candidates": [{key: item[key] for key in ("name", "config", "validation", "selected_epochs", "kappa_m2_day")} for item in physics_search["candidates"]],
                    "evaluation": physics_search["selected_model_evaluation"]},
        "data_handling": {"core_cells": int(core.sum()), "core_rule": "At least 80% coverage of unique training-window maps.",
                          "origin_and_target": "Measured pairs only; no map or target gap filling.",
                          "older_predictors": "Training medians and explicit missingness indicators.",
                          "drivers": "Strictly before origin; 80% daily coverage; no future observed drivers.",
                          "events": "Primary scores retain future deepening/dredging; flags are historical depth-change heuristics.",
                          "calendar": "Daily reanalysis and date-only alignment underpin the backtest; a live driver feed was not evaluated."},
    }
    write_json(args.output / "summary.json", report)
    validation_rows = [{"candidate": row["id"], "validation_mae_ft": row["validation"]["mae_ft"],
                        "training_rows": row["training_rows"], "features": row["features"], "boosting_rounds": row["boosting_rounds"]} for row in xgb_search["candidates"]]
    write_csv(args.output / "validation_candidates.csv", validation_rows)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    names = ["incumbent", "xgboost", "physics_original", "physics", "persistence"]
    axes[0].barh([LABELS[name] for name in names], [summaries[name]["mae_ft"] for name in names], color=[COLORS[name] for name in names])
    axes[0].invert_yaxis(); axes[0].set_xlabel("2025–26 backtest MAE (ft), lower is better")
    for index, name in enumerate(names):
        axes[0].text(summaries[name]["mae_ft"] + .008, index, f'{summaries[name]["mae_ft"]:.3f}', va="center")
    axes[0].set_xlim(0, .82); axes[0].set_title("Final models and retrained baselines")
    for name in ("xgboost", "physics", "persistence"):
        values = [row["mae_ft"] for row in rows if row["model"] == name]
        for year in ("2025", "2026"):
            selected_dates = np.char.startswith(dates, year)
            axes[1].plot(np.asarray(dates[selected_dates], dtype="datetime64[D]"), np.asarray(values)[selected_dates],
                         label=LABELS[name] if year == "2025" else None, color=COLORS[name], linewidth=1.2)
    axes[1].set_ylabel("Per-survey MAE (ft)"); axes[1].set_title("71 forecast dates; observed cells only")
    axes[1].legend(frameon=False); axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle("28-day Anchorage forecasts · validation-selected models · reused backtest", fontsize=13)
    fig.savefig(args.output / "performance.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    ordered = sorted(validation_rows, key=lambda row: row["validation_mae_ft"])
    axes[0].barh([row["candidate"] for row in ordered], [row["validation_mae_ft"] for row in ordered], color="#2166ac")
    axes[0].invert_yaxis(); axes[0].set_xlim(0, .65); axes[0].set_xlabel("Validation MAE (ft)"); axes[0].set_title("XGBoost training ablations")
    physics_candidates = report["physics"]["candidates"]
    axes[1].barh([row["name"] for row in physics_candidates], [row["validation"]["mae_ft"] for row in physics_candidates], color="#bd6c29")
    axes[1].invert_yaxis(); axes[1].set_xlim(0, .75); axes[1].set_xlabel("Validation MAE (ft)"); axes[1].set_title("Physics training-loss ablations")
    fig.suptitle("Model choices use 2023–24 validation only", fontsize=13)
    fig.savefig(args.output / "validation.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 6), constrained_layout=True)
    extent = [0, core.shape[1] * 10, 0, core.shape[0] * 10]
    last_window = windows[split == "test"][-1]
    measured = np.isfinite((archive.maps[last_window[-1]] - archive.maps[last_window[-2]])[core])
    actual_map = np.full(core.shape, np.nan)
    actual_map[core] = (archive.maps[last_window[-1]] - archive.maps[last_window[-2]])[core] / FEET_TO_METRES
    panels = [actual_map]
    for name in ("xgboost", "physics"):
        prediction_map = np.full(core.shape, np.nan); prediction_map[core] = np.where(measured, latest[name], np.nan)
        panels.append(prediction_map)
    for ax, values, title in zip(axes, panels, ("Observed change", "Final XGBoost", "Final physics")):
        mesh = ax.imshow(values, origin="lower", extent=extent, cmap="RdBu", vmin=-3, vmax=3, interpolation="nearest")
        ax.set_title(title); ax.set_xlabel("Easting from grid origin (m)"); ax.set_ylabel("Northing from grid origin (m)")
    fig.colorbar(mesh, ax=axes, label="28-day depth change (ft) · negative = shallower", shrink=.8, extend="both")
    fig.suptitle(f'{archive.dates[last_window[-2]]} → {archive.dates[last_window[-1]]} · common observed origin/target pairs', fontsize=12)
    fig.savefig(args.output / "change_maps.png", dpi=170); plt.close(fig)
    print(json.dumps({"output": str(args.output), "test": summaries, "paired_comparison": comparisons}, indent=2))


if __name__ == "__main__":
    main()
