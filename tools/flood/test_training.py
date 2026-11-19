import argparse
import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from training_data import fit_scalers, load_arrays, partition, read_manifest, validate_files
from train import ForecastDataset, FloodForecaster, evaluate, train


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.csv"
        self.rows = []
        for year in (2017, 2019, 2021):
            for index in range(4):
                issued = datetime(year, 3, 1, tzinfo=timezone.utc) + timedelta(days=index)
                identifier = f"{year}-{index}"
                row = {"sample_id": identifier, "event_group": identifier, "region": "example",
                       "issued_at": issued.isoformat(), "history_start": (issued - timedelta(days=22)).isoformat(),
                       "target_start": (issued + timedelta(days=5)).isoformat(),
                       "target_end": (issued + timedelta(days=6)).isoformat(),
                       "label": index % 2, "npz_path": f"{identifier}.npz"}
                precip_times = [issued - timedelta(days=1) - timedelta(hours=3 * step) for step in reversed(range(32))]
                soil_times = [issued - timedelta(days=3 + step) for step in reversed(range(14))]
                # Deliberately larger held-out inputs reveal accidental scaler fitting.
                precip = np.full((32, 8, 8), 1 + index + (100 if year > 2017 else 0), dtype=np.float32)
                soil = np.full((14, 8, 8), 0.2 + index * 0.01, dtype=np.float32)
                soil[0, 0, 0] = np.nan
                np.savez(self.root / row["npz_path"], precip=precip, soil=soil,
                         precip_times=[value.isoformat() for value in precip_times],
                         soil_times=[value.isoformat() for value in soil_times],
                         precip_available_at=[(value + timedelta(hours=12)).isoformat() for value in precip_times],
                         soil_available_at=[(value + timedelta(days=3)).isoformat() for value in soil_times])
                self.rows.append(row)
        with self.manifest.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.parsed = read_manifest(self.manifest)
        self.splits, _ = partition(self.parsed, "2019-01-01T00:00:00Z", "2021-01-01T00:00:00Z", 5)

    def tearDown(self):
        self.temp.cleanup()

    def test_training_only_scalers_and_deterministic_validation(self):
        scalers = fit_scalers(self.splits["train"])
        self.assertEqual(float(scalers["precip"].data_max_[0]), 4)
        self.assertLess(float(scalers["soil"].data_max_[0]), 1)
        dataset = ForecastDataset(self.splits["validation"], scalers, augment=False)
        first, second = dataset[0], dataset[0]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        self.assertGreater(float(first[0].max()), 1)  # Held-out extremes aren't clipped away.
        self.assertEqual(float(first[1][0, 1, 0, 0]), 0)  # Missing-soil mask survives.

    def test_future_release_is_rejected(self):
        row = self.parsed[0]
        with np.load(row["npz_path"]) as source:
            arrays = {key: source[key] for key in source.files}
        arrays["precip_available_at"][:] = "2099-01-01T00:00:00+00:00"
        np.savez(row["npz_path"], **arrays)
        with self.assertRaisesRegex(ValueError, "published"):
            load_arrays(row)

    def test_missing_daily_step_is_rejected(self):
        row = self.parsed[0]
        with np.load(row["npz_path"]) as source:
            arrays = {key: source[key] for key in source.files}
        arrays["soil_times"][2] = arrays["soil_times"][1]
        np.savez(row["npz_path"], **arrays)
        with self.assertRaisesRegex(ValueError, "cadence"):
            load_arrays(row)

    def test_event_group_cannot_cross_partitions(self):
        self.parsed[0]["event_group"] = self.parsed[4]["event_group"] = "one-storm"
        splits, excluded = partition(self.parsed, "2019-01-01T00:00:00Z", "2021-01-01T00:00:00Z", 5)
        self.assertEqual(sum(row["event_group"] == "one-storm" for rows in splits.values() for row in rows), 0)
        self.assertEqual(len(excluded), 2)

    def test_full_history_window_is_purged_at_boundary(self):
        self.parsed[4]["history_start"] = datetime(2018, 12, 31, tzinfo=timezone.utc)
        _, excluded = partition(self.parsed, "2019-01-01T00:00:00Z", "2021-01-01T00:00:00Z", 5)
        self.assertEqual(excluded[0]["reason"], "window_crosses_partition")

    def test_identical_files_in_different_partitions_are_rejected(self):
        self.splits["test"][0]["npz_path"] = self.splits["train"][0]["npz_path"]
        with self.assertRaisesRegex(ValueError, "Identical"):
            validate_files(self.splits)

    def test_region_holdout_excludes_region_from_training_and_validation(self):
        for row in self.parsed[8:]:
            row["region"] = "holdout"
        self.parsed[0]["region"] = self.parsed[4]["region"] = "holdout"
        self.parsed[8]["region"] = "other"
        splits, excluded = partition(self.parsed, "2019-01-01T00:00:00Z", "2021-01-01T00:00:00Z", 5, ("holdout",))
        self.assertTrue(all(row["region"] != "holdout" for name in ("train", "validation") for row in splits[name]))
        self.assertTrue(all(row["region"] == "holdout" for row in splits["test"]))
        self.assertEqual(len(excluded), 3)

    def test_wrong_forecast_lead_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "forecast lead"):
            partition(self.parsed, "2019-01-01T00:00:00Z", "2021-01-01T00:00:00Z", 7)

    def test_encoder_uses_final_precipitation_steps(self):
        torch.manual_seed(42)
        model = FloodForecaster().eval()
        p = torch.ones(1, 32, 1, 8, 8)
        s = torch.ones(1, 14, 2, 8, 8)
        ages_p, ages_s = torch.zeros(1, 32), torch.zeros(1, 14)
        with torch.no_grad():
            first = model(p, s, ages_p, ages_s)
            p[:, -1] = 50
            second = model(p, s, ages_p, ages_s)
        self.assertFalse(torch.equal(first, second))

    def test_end_to_end_training_then_separate_test_evaluation(self):
        output = self.root / "run"
        args = argparse.Namespace(manifest=self.manifest, output=output, validation_start="2019-01-01T00:00:00Z",
                                  test_start="2021-01-01T00:00:00Z", lead_days=5, test_regions="", epochs=2,
                                  patience=1, batch_size=2, learning_rate=0.001, seed=42, threads=1,
                                  device="cpu", augment=True)
        train(args)
        self.assertFalse((output / "test_metrics.json").exists())
        self.assertFalse(json.loads((output / "training.json").read_text())["test_evaluated"])
        evaluate(argparse.Namespace(manifest=self.manifest, checkpoint=output, batch_size=2))
        report = json.loads((output / "test_metrics.json").read_text())
        self.assertEqual(report["test"]["examples"], 4)
        self.assertIn("precision_recall_curve", report["test"])
        self.assertIn("training_prior_baseline", report)


if __name__ == "__main__":
    unittest.main()
