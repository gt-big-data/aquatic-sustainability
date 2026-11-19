"""Explicit forecast-time contract and leakage-resistant flood partitions."""
import csv
import hashlib
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


def timestamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("All timestamps must include a timezone, preferably Z/UTC")
    return result.astimezone(timezone.utc)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    required = {"sample_id", "event_group", "region", "issued_at", "history_start", "target_start", "target_end", "label", "npz_path"}
    seen = set()
    for row in rows:
        if not required.issubset(row) or any(not str(row[key]).strip() for key in required):
            raise ValueError(f"Manifest requires nonempty columns: {sorted(required)}")
        if row["sample_id"] in seen:
            raise ValueError("sample_id must be unique")
        seen.add(row["sample_id"])
        if row["label"] not in {"0", "1"}:
            raise ValueError("Labels must be 0 or 1")
        row["label"] = int(row["label"])
        for key in ("issued_at", "history_start", "target_start", "target_end"):
            row[key] = timestamp(row[key])
        if not row["history_start"] < row["issued_at"] < row["target_start"] <= row["target_end"]:
            raise ValueError("Require history_start < issued_at < target_start <= target_end")
        row["npz_path"] = str((path.parent / row["npz_path"]).resolve())
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


def partition(rows, validation_start, test_start, lead_days, test_regions=()):
    validation_start, test_start = timestamp(validation_start), timestamp(test_start)
    if validation_start >= test_start or lead_days <= 0:
        raise ValueError("Require positive lead and validation_start < test_start")
    splits = {name: [] for name in ("train", "validation", "test")}
    dropped = []
    for row in rows:
        if row["target_start"] - row["issued_at"] != timedelta(days=lead_days):
            raise ValueError(f"Sample {row['sample_id']} does not match the declared forecast lead")
        if row["target_end"] < validation_start:
            split = "train"
        elif row["history_start"] >= validation_start and row["target_end"] < test_start:
            split = "validation"
        elif row["history_start"] >= test_start:
            split = "test"
        else:
            dropped.append({"sample_id": row["sample_id"], "reason": "window_crosses_partition"})
            continue
        if test_regions and ((row["region"] in test_regions) != (split == "test")):
            dropped.append({"sample_id": row["sample_id"], "reason": "region_holdout"})
            continue
        splits[split].append(row)
    groups = defaultdict(set)
    for split, samples in splits.items():
        for row in samples:
            groups[row["event_group"]].add(split)
    crossing = {group for group, names in groups.items() if len(names) > 1}
    for split in splits:
        for row in splits[split]:
            if row["event_group"] in crossing:
                dropped.append({"sample_id": row["sample_id"], "reason": "event_group_crosses_partition"})
        splits[split] = [row for row in splits[split] if row["event_group"] not in crossing]
        if {row["label"] for row in splits[split]} != {0, 1}:
            raise ValueError(f"{split} must contain both classes after purging")
    return splits, dropped


def validate_files(splits):
    """Reject the same prepared input being reused across partitions."""
    hashes, locations = {}, {}
    for split, rows in splits.items():
        for row in rows:
            digest = file_hash(row["npz_path"])
            if digest in locations and locations[digest] != split:
                raise ValueError("Identical NPZ inputs occur in different partitions")
            locations[digest] = split
            hashes[row["sample_id"]] = digest
    return hashes


def load_arrays(row, precip_steps=32, soil_steps=14):
    """No skipped days, implicit padding, future measurements, or future releases."""
    arrays, ages = [], []
    with np.load(row["npz_path"], allow_pickle=False) as source:
        for name, length, cadence in (("precip", precip_steps, 3), ("soil", soil_steps, 24)):
            values = np.asarray(source[name], dtype=np.float32)
            if values.ndim != 3 or values.shape[0] != length or min(values.shape[1:]) < 8:
                raise ValueError(f"{name} must have shape ({length}, H>=8, W>=8)")
            times = [timestamp(value) for value in source[f"{name}_times"].astype(str)]
            available = [timestamp(value) for value in source[f"{name}_available_at"].astype(str)]
            if len(times) != length or len(available) != length:
                raise ValueError("Each frame requires its interval-end and availability timestamp")
            if any(b - a != timedelta(hours=cadence) for a, b in zip(times, times[1:])):
                raise ValueError(f"{name} cadence must be {cadence} hours; missing frames cannot be compressed")
            if times[0] < row["history_start"] or times[-1] > row["issued_at"]:
                raise ValueError("Measurements must lie inside the declared pre-issuance history")
            if any(released < measured or released > row["issued_at"] for measured, released in zip(times, available)):
                raise ValueError("Every input must have been published by forecast issuance")
            if np.isinf(values).any() or any(not np.isfinite(frame).any() for frame in values):
                raise ValueError("Infinite data or an entirely missing frame is not allowed")
            if name == "precip" and (not np.isfinite(values).all() or (values < 0).any()):
                raise ValueError("Precipitation accumulation must be finite and nonnegative")
            if name == "soil" and ((values[np.isfinite(values)] < 0).any() or (values[np.isfinite(values)] > 1).any()):
                raise ValueError("Soil moisture must be a volumetric fraction between 0 and 1")
            arrays.append(values)
            ages.append(np.array([(row["issued_at"] - measured).total_seconds() / 86400 for measured in times], dtype=np.float32))
    return (*arrays, *ages)


def fit_scalers(rows):
    """Separate streaming scalers for the two physical quantities; train rows only."""
    from sklearn.preprocessing import MinMaxScaler

    scalers = {"precip": MinMaxScaler(), "soil": MinMaxScaler()}
    for row in rows:
        precip, soil, _, _ = load_arrays(row)
        for name, values in (("precip", precip), ("soil", soil)):
            finite = values[np.isfinite(values)]
            scalers[name].partial_fit(finite.reshape(-1, 1))
    return scalers
