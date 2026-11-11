"""Train the XGBoost neighborhood adapter using this branch's input pipeline."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from forecast import CACHE, DEFAULT_MODEL, atomic_write, evaluate, fit_model, pinned_dataset, read_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    dataset = pinned_dataset(args.dataset)
    archive = read_archive(dataset)
    with np.load(dataset, allow_pickle=False) as source:
        windows, split = source["windows"], source["split"]
    if windows.ndim != 2 or windows.shape[1] != 5 or split.shape != (len(windows),):
        raise ValueError("Expected aligned four-map histories, targets and split labels.")
    if not (
        archive.dates[windows[split == "train", -1]].max() < archive.dates[windows[split == "val", 0]].min()
        and archive.dates[windows[split == "val", -1]].max() < archive.dates[windows[split == "test", 0]].min()
    ):
        raise ValueError("Training, validation and test windows must be chronologically disjoint.")
    model = fit_model(archive, windows, split)
    model["evaluation"] = {
        key: evaluate(model, archive, windows[split == ("val" if key == "validation" else key)]) for key in ("validation", "test")
    }
    atomic_write(args.output, lambda path: joblib.dump(model, path, compress=3))
    print(json.dumps({"artifact": str(args.output), "model": model["model_name"],
                      "evaluation": model["evaluation"]}, indent=2))


if __name__ == "__main__":
    main()
