"""Train the physics adapter with its own survey and environmental inputs."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from forecast import CACHE, atomic_write, pinned_dataset, read_archive
from physics import train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=CACHE / "physics.joblib")
    args = parser.parse_args()
    dataset = pinned_dataset(args.dataset)
    archive = read_archive(dataset)
    with np.load(dataset, allow_pickle=False) as source:
        windows, split = source["windows"], source["split"]
    model = train_model(archive, dataset, windows, split)
    atomic_write(args.output, lambda path: joblib.dump(model, path, compress=3))
    print(json.dumps({"artifact": str(args.output), "model": model["model_name"],
                      "evaluation": model["evaluation"]}, indent=2))


if __name__ == "__main__":
    main()
