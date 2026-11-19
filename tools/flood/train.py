"""Train a new flood research model; never overwrite the website's V3/V9 files."""
import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, precision_recall_curve,
                             precision_score, recall_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training_data import file_hash, fit_scalers, load_arrays, partition, read_manifest, validate_files


class ForecastDataset(Dataset):
    def __init__(self, rows, scalers, augment=False, expected_shapes=None):
        self.rows, self.scalers, self.augment = rows, scalers, augment
        self.expected_shapes = expected_shapes

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        precip, soil, precip_ages, soil_ages = load_arrays(self.rows[index])
        if self.expected_shapes and (list(precip.shape) != self.expected_shapes[0] or list(soil.shape) != self.expected_shapes[1]):
            raise ValueError("All crops must match the training grid dimensions")
        mask = np.isfinite(soil)
        p = self.scalers["precip"].transform(precip.reshape(-1, 1)).reshape(precip.shape).astype(np.float32)
        # Fill missing soil cells after scaling; an explicit mask distinguishes them.
        s = np.zeros_like(soil)
        s[mask] = self.scalers["soil"].transform(soil[mask].reshape(-1, 1)).ravel()
        if self.augment:
            p = p + np.random.normal(0, 0.02, p.shape).astype(np.float32)
            s[mask] += np.random.normal(0, 0.02, int(mask.sum())).astype(np.float32)
        return (torch.from_numpy(p[:, None]), torch.from_numpy(np.stack([s, mask.astype(np.float32)], axis=1)),
                torch.from_numpy(precip_ages), torch.from_numpy(soil_ages),
                torch.tensor(float(self.rows[index]["label"])))


class FloodForecaster(nn.Module):
    """Independent temporal encoders: do not pair a 3-hour frame with a day by index."""
    def __init__(self):
        super().__init__()
        def encoder(channels):
            return nn.Sequential(nn.Conv2d(channels, 8, 3, 2, 1), nn.ReLU(),
                                 nn.Conv2d(8, 16, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((2, 2)))
        self.precip_cnn, self.soil_cnn = encoder(1), encoder(2)
        self.precip_lstm, self.soil_lstm = nn.LSTM(65, 32, batch_first=True), nn.LSTM(65, 32, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(64, 1))

    def forward(self, precip, soil, precip_ages, soil_ages):
        def summarize(values, ages, cnn, lstm):
            batch, steps, channels, height, width = values.shape
            features = cnn(values.reshape(batch * steps, channels, height, width)).reshape(batch, steps, 64)
            # Fixed conversion to weeks; no preprocessing fitted to held-out periods.
            features = torch.cat([features, ages[..., None] / 7], dim=-1)
            _, (hidden, _) = lstm(features)
            return hidden[-1]
        joined = torch.cat([summarize(precip, precip_ages, self.precip_cnn, self.precip_lstm),
                            summarize(soil, soil_ages, self.soil_cnn, self.soil_lstm)], dim=1)
        return self.head(joined).squeeze(1)


def metrics(labels, probabilities, threshold=0.5):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    predictions = probabilities >= threshold
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    return {"examples": len(labels), "positive_fraction": float(labels.mean()), "threshold": threshold,
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "average_precision": float(average_precision_score(labels, probabilities)),
            "brier_score": float(np.mean((probabilities - labels) ** 2)),
            "confusion_matrix_tn_fp_fn_tp": confusion_matrix(labels, predictions, labels=[0, 1]).ravel().tolist(),
            "precision_recall_curve": {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": thresholds.tolist()}}


def score(model, loader, device):
    model.eval()
    labels, probabilities, losses = [], [], []
    with torch.no_grad():
        for *inputs, target in loader:
            target = target.to(device)
            logits = model(*(values.to(device) for values in inputs))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="sum")
            losses.append(float(loss))
            labels.extend(target.cpu().tolist())
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    return sum(losses) / len(labels), labels, probabilities


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def split_report(splits, dropped):
    return {"samples": {name: [row["sample_id"] for row in rows] for name, rows in splits.items()},
            "event_groups": {name: sorted({row["event_group"] for row in rows}) for name, rows in splits.items()},
            "label_counts": {name: {str(label): sum(row["label"] == label for row in rows) for label in (0, 1)} for name, rows in splits.items()},
            "excluded": dropped}


def train(args):
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Training bounds must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new output directory for each experiment")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rows = read_manifest(args.manifest)
    regions = tuple(item.strip() for item in args.test_regions.split(",") if item.strip())
    splits, dropped = partition(rows, args.validation_start, args.test_start, args.lead_days, regions)
    hashes = validate_files(splits)
    scalers = fit_scalers(splits["train"])
    first = load_arrays(splits["train"][0])
    shapes = [list(first[0].shape), list(first[1].shape)]
    training = ForecastDataset(splits["train"], scalers, augment=args.augment, expected_shapes=shapes)
    validation = ForecastDataset(splits["validation"], scalers, augment=False, expected_shapes=shapes)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model = FloodForecaster().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history, best, best_epoch, best_weights, stale = [], float("inf"), 0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for *inputs, target in loader:
            target = target.to(device)
            optimizer.zero_grad()
            logits = model(*(values.to(device) for values in inputs))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(target)
        val_loss, _, _ = score(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": total / len(training), "validation_loss": val_loss})
        print(json.dumps(history[-1]), flush=True)
        if val_loss < best:
            best, best_epoch, stale = val_loss, epoch, 0
            best_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_weights)
    _, labels, probabilities = score(model, val_loader, device)
    config = {"validation_start": args.validation_start, "test_start": args.test_start,
              "lead_days": args.lead_days, "test_regions": regions}
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(scalers, args.output / "scalers.joblib")
    torch.save({"architecture": "separate-temporal-encoders-v1", "state_dict": best_weights,
                "manifest_sha256": file_hash(args.manifest), "input_sha256": hashes,
                "scalers_sha256": file_hash(args.output / "scalers.joblib"), "expected_shapes": shapes,
                "partition": config, "best_epoch": best_epoch, "seed": args.seed,
                "threshold": 0.5, "training_positive_fraction": float(np.mean([row["label"] for row in splits["train"]]))}, args.output / "model.pt")
    write_json(args.output / "splits.json", split_report(splits, dropped))
    write_json(args.output / "training.json", {"config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                                               "best_epoch": best_epoch, "history": history,
                                               "validation": metrics(labels, probabilities), "test_evaluated": False})
    print("Training complete. Test scores were not computed; freeze the recipe before evaluate.")


def evaluate(args):
    # Load only artifacts produced by this trainer, not downloaded arbitrary pickles.
    checkpoint = torch.load(args.checkpoint / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint["manifest_sha256"] != file_hash(args.manifest):
        raise ValueError("Evaluation must use the frozen training manifest")
    splits, _ = partition(read_manifest(args.manifest), **checkpoint["partition"])
    if validate_files(splits) != checkpoint["input_sha256"]:
        raise ValueError("Prepared input files changed after training")
    if file_hash(args.checkpoint / "scalers.joblib") != checkpoint["scalers_sha256"]:
        raise ValueError("Scalers changed after training")
    scalers = joblib.load(args.checkpoint / "scalers.joblib")
    dataset = ForecastDataset(splits["test"], scalers, augment=False, expected_shapes=checkpoint["expected_shapes"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = FloodForecaster()
    model.load_state_dict(checkpoint["state_dict"])
    loss, labels, probabilities = score(model, loader, torch.device("cpu"))
    baseline = np.full(len(labels), checkpoint["training_positive_fraction"])
    report = {"model_sha256": file_hash(args.checkpoint / "model.pt"), "test_loss": loss,
              "test": metrics(labels, probabilities, checkpoint["threshold"]),
              "training_prior_baseline": metrics(labels, baseline, checkpoint["threshold"]),
              "predictions": [{"sample_id": row["sample_id"], "label": label, "probability": probability}
                              for row, label, probability in zip(splits["test"], labels, probabilities)]}
    write_json(args.checkpoint / "test_metrics.json", report)
    print(json.dumps({key: value for key, value in report["test"].items() if key != "precision_recall_curve"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--manifest", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--validation-start", required=True)
    fit.add_argument("--test-start", required=True)
    fit.add_argument("--lead-days", type=int, required=True)
    fit.add_argument("--test-regions", default="")
    fit.add_argument("--epochs", type=int, default=100)
    fit.add_argument("--patience", type=int, default=10)
    fit.add_argument("--batch-size", type=int, default=4)
    fit.add_argument("--learning-rate", type=float, default=1e-4)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--threads", type=int, default=4)
    fit.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    fit.add_argument("--augment", action="store_true")
    test = commands.add_parser("evaluate")
    test.add_argument("--manifest", type=Path, required=True)
    test.add_argument("--checkpoint", type=Path, required=True)
    test.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    train(args) if args.command == "train" else evaluate(args)


if __name__ == "__main__":
    main()
