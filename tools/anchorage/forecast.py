"""XGBoost input preparation, training and inference; publish via publish_overlays.py."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import numpy as np
import sklearn
from scipy.ndimage import uniform_filter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from candidates import PROFILES, fit_estimator, predict


SOURCE_COMMIT = "c9e6f432b581aea2fd435618f72522ca7f784167"
SOURCE_PATH = "anchorage/data/timeseries_v2/timeseries.npz"
SOURCE_URL = f"https://raw.githubusercontent.com/gt-big-data/aquatic-sustainability/{SOURCE_COMMIT}/{SOURCE_PATH}"
SOURCE_BLOB = "7b603395ef4992b201bc3dc00ef8c3ca1a22c521"
HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
DEFAULT_MODEL = CACHE / "xgboost.joblib"
RELEASE_PROFILE = "xgb-spatial"
FEET_TO_METRES = 0.3048
LEAD_DAYS = 28
SCHEMA_VERSION = 1
TRAIN_CELLS_PER_WINDOW = 400
RANDOM_SEED = 42
MAP_FEATURE_NAMES = (
    "row", "col", "last_depth_m", "history0_minus_last_m", "history1_minus_last_m",
    "history2_minus_last_m", "historical_rate_m_28d", "gap_0_days", "gap_1_days",
    "gap_2_days", "month_sin", "month_cos", "past_dredging_flag", "baseline28_m",
)
NEIGHBOR_FEATURE_NAMES = tuple(
    f"{name}_{width}m"
    for width in (30, 90, 210)
    for name in ("depth_anomaly", "roughness", "recent_rate", "longer_rate", "recent_coverage")
)


@dataclass
class Archive:
    maps: np.ndarray
    dates: np.ndarray
    dredged: np.ndarray
    grid: dict
    step: int


def read_archive(path):
    with np.load(path, allow_pickle=False) as source:
        maps = source["maps"].astype(np.float64) * FEET_TO_METRES
        dates = source["dates"].astype("datetime64[D]")
        dredged = source["dredged"]
        step = int(source["STEP"])
        if maps.ndim != 3 or len(maps) != len(dates) or dredged.shape != maps.shape:
            raise ValueError("Survey dates, depth maps, and event flags must align.")
        if dates.ndim != 1 or len(dates) == 0 or np.isnat(dates).any() or (dates[1:] <= dates[:-1]).any():
            raise ValueError("Survey dates must be valid, distinct, and chronological.")
        if min(maps.shape[1:]) < 2 or dredged.dtype != np.bool_ or int(source["K"]) != 4 or step != 2:
            raise ValueError("Expected a v2 grid, boolean event flags, and four maps at stride two.")
        grid = {
            "rows": maps.shape[1], "cols": maps.shape[2],
            "row0": int(source["row0"]), "col0": int(source["col0"]),
            "cell_metres": int(source["cell"]), "crs": "EPSG:26934",
        }
    if grid["cell_metres"] != 10:
        raise ValueError("The benchmark uses a 10 m grid.")
    return Archive(maps, dates, dredged, grid, step)


def source_hash(path):
    contents = path.read_bytes()
    return hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()


def pinned_dataset(path):
    if path is None:
        path = CACHE / "timeseries.npz"
        if not path.exists():
            CACHE.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=CACHE) as directory:
                download = Path(directory) / path.name
                urlretrieve(SOURCE_URL, download)
                if source_hash(download) != SOURCE_BLOB:
                    raise ValueError("Downloaded dataset differs from the pinned GitHub blob.")
                download.replace(path)
    if source_hash(path) != SOURCE_BLOB:
        raise ValueError("Training requires the pinned Anchorage v2 dataset.")
    return path


def laplacian(depth, observed, spacing):
    result = np.zeros_like(depth)
    for first, second in [(np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])]:
        shared = observed[first] & observed[second]
        exchange = np.where(shared, depth[second] - depth[first], 0) / spacing**2
        result[first] += exchange
        result[second] -= exchange
    return result


def neighborhood_features(archive, core, history, gaps):
    first, _, previous, last = archive.maps[history]
    observed = core & np.isfinite(last)
    recent = (last - previous) * LEAD_DAYS / gaps[-1]
    longer = (last - first) * LEAD_DAYS / gaps.sum()
    recent_observed = observed & np.isfinite(recent)
    longer_observed = observed & np.isfinite(longer)
    columns = []

    def local_mean(values, mask, width):
        coverage = uniform_filter(mask.astype(float), size=width, mode="constant")
        total = uniform_filter(np.where(mask, values, 0.0), size=width, mode="constant")
        mean = np.divide(total, coverage, out=np.full_like(total, np.nan), where=coverage > 1e-10)
        return mean, coverage

    for width in (3, 9, 21):
        mean_depth, _ = local_mean(last, observed, width)
        mean_square, _ = local_mean(last**2, observed, width)
        roughness = np.sqrt(np.maximum(0, mean_square - mean_depth**2))
        recent_mean, recent_coverage = local_mean(recent, recent_observed, width)
        long_mean, _ = local_mean(longer, longer_observed, width)
        columns.extend([(mean_depth - last)[core], roughness[core], recent_mean[core],
                        long_mean[core], recent_coverage[core]])
    return np.column_stack(columns).astype(np.float32)


def raw_features(archive, core, baseline, history, spatial=False):
    rows, cols = np.where(core)
    maps = archive.maps[history][:, core]
    gaps = np.diff(archive.dates[history]).astype(int)
    current = history[-1]
    month = archive.dates[current].astype("datetime64[M]").astype(int) % 12 + 1
    size = len(rows)
    last = maps[-1]
    rate = (last - maps[0]) * LEAD_DAYS / gaps.sum()
    past_flags = archive.dredged[history[0] + 1:current + 1][:, core].any(axis=0)
    raw = np.column_stack([
        rows / (core.shape[0] - 1), cols / (core.shape[1] - 1), last,
        *(maps[index] - last for index in range(3)), rate,
        *(np.full(size, gap) for gap in gaps),
        np.full(size, np.sin(month * np.pi / 6)), np.full(size, np.cos(month * np.pi / 6)),
        past_flags, baseline,
    ]).astype(np.float32)
    depth = archive.maps[current]
    curvature = laplacian(depth, core & np.isfinite(depth), archive.grid["cell_metres"])[core]
    if spatial:
        raw = np.column_stack([raw, neighborhood_features(archive, core, history, gaps)])
    return raw, curvature, np.isfinite(last)


def design_matrix(model, raw, curvature, lead):
    scaled = model["scaler"].transform(model["imputer"].transform(raw))
    return np.column_stack([
        scaled, np.isnan(raw), curvature, np.broadcast_to(np.asarray(lead) / LEAD_DAYS, len(raw)),
    ]).astype(np.float32)


def observed_changes(archive, core, window):
    origin, target = window[-2:]
    return (archive.maps[target] - archive.maps[origin])[core]


def training_data(archive, windows, split, spatial=False):
    training = windows[split == "train"]
    core = np.isfinite(archive.maps[np.unique(training)]).mean(axis=0) >= 0.8
    lead = (archive.dates[training[:, -1]] - archive.dates[training[:, -2]]).astype(int)
    net_change = np.zeros(int(core.sum()))
    observed_days = np.zeros_like(net_change)
    for window, days in zip(training, lead):
        change = observed_changes(archive, core, window)
        valid = np.isfinite(change)
        net_change[valid] += change[valid]
        observed_days[valid] += days
    average_rate = net_change.sum() / observed_days.sum()
    prior_days = 20 * LEAD_DAYS
    baseline = LEAD_DAYS * (net_change + prior_days * average_rate) / (observed_days + prior_days)

    raw_rows, curvatures, changes, horizons, weights = [], [], [], [], []
    rng = np.random.default_rng(RANDOM_SEED)
    for window, days in zip(training, lead):
        change = observed_changes(archive, core, window)
        valid = np.flatnonzero(np.isfinite(change))
        size = min(TRAIN_CELLS_PER_WINDOW, len(valid))
        if size == 0:
            continue
        chosen = np.sort(rng.choice(valid, size, replace=False))
        raw, curvature, _ = raw_features(archive, core, baseline, window[:-1], spatial)
        raw_rows.append(raw[chosen])
        curvatures.append(curvature[chosen])
        changes.append(change[chosen])
        horizons.append(np.full(size, days))
        weights.append(np.full(size, 1 / size))

    raw = np.concatenate(raw_rows)
    sample_weights = np.concatenate(weights)
    sample_weights /= sample_weights.mean()
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(raw)
    scaler = StandardScaler().fit(imputer.transform(raw))
    model = {
        "version": SCHEMA_VERSION, "sklearn": sklearn.__version__, "source_commit": SOURCE_COMMIT,
        "core": core, "grid": archive.grid, "baseline": baseline,
        "imputer": imputer, "scaler": scaler, "spatial": spatial,
        "training_rows": len(raw), "train_end": str(archive.dates[training[:, -1]].max()),
        "max_history_gap": int(np.diff(archive.dates[training[:, :-1]], axis=1).astype(int).max()),
        "target_months": sorted(set(archive.dates[windows[split != "train", -1]].astype("datetime64[M]").astype(int) % 12 + 1)),
    }
    return (model, design_matrix(model, raw, np.concatenate(curvatures), np.concatenate(horizons)),
            np.concatenate(changes), sample_weights)


def fit_model(archive, windows, split):
    profile = PROFILES[RELEASE_PROFILE]
    model, features, changes, weights = training_data(archive, windows, split, profile.spatial)
    validation_features, validation_changes, counts = evaluation_inputs(model, archive, windows[split == "val"])
    validation_weights = np.repeat(1 / counts, counts)
    validation_weights /= validation_weights.mean()
    with threadpool_limits(limits=4):
        model["estimator"] = fit_estimator(
            RELEASE_PROFILE, features, changes, weights,
            (validation_features, validation_changes, validation_weights),
        )
    model["model_name"] = profile.family
    model["profile_id"] = RELEASE_PROFILE
    names = MAP_FEATURE_NAMES + (NEIGHBOR_FEATURE_NAMES if profile.spatial else ())
    model["input_features"] = list(names) + [f"{name}_missing" for name in names] + ["laplacian_m_per_m2", "lead_over_28d"]
    model["training_windows"] = int((split == "train").sum())
    return model


def predict_change(model, archive, history, lead=LEAD_DAYS):
    raw, curvature, valid = raw_features(
        archive, model["core"], model["baseline"], history, model.get("spatial", False),
    )
    prediction = np.full(len(valid), np.nan)
    if valid.any():
        prediction[valid] = predict(model["estimator"], design_matrix(model, raw[valid], curvature[valid], lead))
    return prediction


def evaluation_inputs(model, archive, windows):
    matrices, targets, counts = [], [], []
    for window in windows:
        lead = int((archive.dates[window[-1]] - archive.dates[window[-2]]).astype(int))
        truth = observed_changes(archive, model["core"], window)
        raw, curvature, _ = raw_features(archive, model["core"], model["baseline"], window[:-1], model["spatial"])
        observed = np.isfinite(truth)
        matrices.append(design_matrix(model, raw[observed], curvature[observed], lead))
        targets.append(truth[observed])
        counts.append(int(observed.sum()))
    if not counts or min(counts) == 0:
        raise ValueError("Every evaluation window must contain observed origin and target depths.")
    return np.concatenate(matrices), np.concatenate(targets), np.asarray(counts)


def evaluate(model, archive, windows):
    errors, persistence, cell_count = [], [], 0
    for window in windows:
        lead = int((archive.dates[window[-1]] - archive.dates[window[-2]]).astype(int))
        truth = observed_changes(archive, model["core"], window)
        prediction = predict_change(model, archive, window[:-1], lead)
        measured = np.isfinite(truth) & np.isfinite(prediction)
        if measured.any():
            errors.append(np.abs(prediction[measured] - truth[measured]).mean() / FEET_TO_METRES)
            persistence.append(np.abs(truth[measured]).mean() / FEET_TO_METRES)
            cell_count += int(measured.sum())
    return {"mae_ft": float(np.mean(errors)), "persistence_mae_ft": float(np.mean(persistence)),
            "windows": len(errors), "scored_cells": cell_count}


def atomic_write(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        writer(temporary)
        temporary.replace(path)
