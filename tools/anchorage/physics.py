"""Independently train a reduced physics-informed Anchorage depth forecaster.

Depth is positive downward, in metres.  The 28-day historical-rate baseline and a
small learned source are combined with a conservative four-neighbour diffusion
update.  This is a constrained source/diffusion surrogate, not an identified
sediment-transport PINN: currents, sediment properties and dredging rates are not
observed.  The source can explain deposition, erosion, or unrecorded operations.

PyTorch is needed only to train.  Bundles contain plain NumPy weights plus fitted
scikit-learn preprocessing; prediction uses NumPy, with no custom pickled class.
"""

import copy
import time

import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


SOURCE_COMMIT = "c9e6f432b581aea2fd435618f72522ca7f784167"
FEET_TO_METRES = 0.3048
REFERENCE_DAYS = 28
MAX_LEAD_DAYS = 40
KAPPA_MAX = 0.625  # 10**2 / (4 * 40), the transport-only convex-update bound.
TRAIN_CELLS_PER_WINDOW = 400
SEEDS = (42, 43)
EPOCHS = 36
BATCH_SIZE = 8192
DRIVERS = ("tide_range", "knik_flow", "matanuska_flow", "shipcreek_flow", "rain", "snowfall", "temp")
FACTORS = {
    "tide_range": FEET_TO_METRES,
    "knik_flow": 0.028316846592,
    "matanuska_flow": 0.028316846592,
    "shipcreek_flow": 0.028316846592,
    "rain": 1.0,
    "snowfall": 10.0,
    "temp": 1.0,
}
MAP_FEATURES = (
    "row", "col", "last_depth_m", "history0_minus_last_m", "history1_minus_last_m",
    "history2_minus_last_m", "historical_rate_m_28d", "gap_0_days", "gap_1_days",
    "gap_2_days", "month_sin", "month_cos", "past_dredging_flag", "baseline28_m",
)
FEATURE_NAMES = MAP_FEATURES + tuple(
    column
    for driver in DRIVERS
    for width in (7, 28)
    for column in (f"{driver}_past{width}", f"{driver}_past{width}_coverage")
)


def read_drivers(path, expected_dates=None):
    """Read only observed daily series; intentionally ignore all *_ahead arrays."""
    result = {}
    with np.load(path, allow_pickle=False) as source:
        if expected_dates is not None and not np.array_equal(
            source["dates"].astype("datetime64[D]"), expected_dates,
        ):
            raise ValueError("Depth maps and daily-driver archive must describe the same surveys.")
        for name in DRIVERS:
            calendar = source["tide_days" if name == "tide_range" else "days"].astype("datetime64[D]")
            values = source[f"{name}_daily"].copy()
            if calendar.ndim != 1 or values.shape != calendar.shape or np.isnat(calendar).any():
                raise ValueError(f"Daily {name} values and dates must be aligned one-dimensional arrays.")
            if (calendar[1:] <= calendar[:-1]).any():
                raise ValueError(f"Daily {name} dates must be distinct and chronological.")
            result[name] = (calendar, values)
    return result


def driver_features(drivers, origin):
    """Summaries of [origin-width, origin), requiring 80% finite daily coverage.

    Flow is converted cfs -> m3/s then log1p; tide ft -> m; snow cm -> mm.
    Rain/snow means become full-window-equivalent totals only after passing the
    coverage threshold.  Coverage is supplied separately, including for gaps.
    """
    values = []
    for name in DRIVERS:
        calendar, daily = drivers[name]
        for width in (7, 28):
            selected = daily[(calendar >= origin - np.timedelta64(width, "D")) & (calendar < origin)]
            selected = selected * FACTORS[name]
            coverage = np.isfinite(selected).sum() / width
            value = np.nanmean(selected) if coverage >= 0.8 else np.nan
            if name in ("rain", "snowfall"):
                value *= width
            elif name.endswith("_flow"):
                value = np.log1p(value)
            values.extend((value, coverage))
    return np.asarray(values)


def graph_laplacian(depth, observed, spacing=10):
    """Exchange only across observed/core edges, with equal/opposite updates.

    An unobserved neighbour is an absent edge, not a zero-depth observation.
    This closes the masked graph (no flux across its boundary); it is not a
    measured hydrodynamic boundary condition.
    """
    result = np.zeros_like(depth, dtype=float)
    for first, second in ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])):
        shared = observed[first] & observed[second]
        exchange = np.where(shared, depth[second] - depth[first], 0.0) / spacing**2
        result[first] += exchange
        result[second] -= exchange
    return result


def raw_features(archive, core, baseline, history, drivers):
    """Build 42 raw causal predictors and the separate physical Laplacian."""
    history = np.asarray(history, dtype=int)
    if history.shape != (4,) or (np.diff(history) <= 0).any():
        raise ValueError("Physics inference requires four chronological survey indices.")
    if history[0] < 0 or history[-1] >= len(archive.dates):
        raise ValueError("History indices must refer to maps in the supplied archive.")
    rows, cols = np.where(core)
    hist = archive.maps[history][:, core]
    last = hist[-1]
    gaps = np.diff(archive.dates[history]).astype(int)
    if (gaps <= 0).any():
        raise ValueError("History dates must be distinct and chronological.")
    trend = (last - hist[0]) / gaps.sum() * REFERENCE_DAYS
    origin = history[-1]
    month = archive.dates[origin].astype("datetime64[M]").astype(int) % 12 + 1
    count = len(rows)
    driver = driver_features(drivers, archive.dates[origin])
    raw = np.column_stack([
        rows / (core.shape[0] - 1), cols / (core.shape[1] - 1), last,
        *(hist[index] - last for index in range(3)), trend,
        *(np.full(count, gap) for gap in gaps),
        np.full(count, np.sin(month * np.pi / 6)), np.full(count, np.cos(month * np.pi / 6)),
        archive.dredged[history[0] + 1:origin + 1][:, core].any(axis=0), baseline,
        np.broadcast_to(driver, (count, len(driver))),
    ]).astype(np.float32)
    observed = core & np.isfinite(archive.maps[origin])
    curvature = graph_laplacian(archive.maps[origin], observed, archive.grid["cell_metres"])[core]
    return raw, curvature, np.isfinite(last)


def design_matrix(model, raw):
    """Training medians + training standardization + a flag for every predictor."""
    values = model["scaler"].transform(model["imputer"].transform(raw))
    features = np.column_stack([values, np.isnan(raw).astype(float)]).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Physics features must be finite after training-only preprocessing.")
    return features


def _calibration(archive, windows):
    core = np.isfinite(archive.maps[np.unique(windows)]).mean(axis=0) >= 0.8
    if not core.any():
        raise ValueError("No cells meet the 80% training-only survey coverage threshold.")
    total_delta = np.zeros(int(core.sum()))
    total_days = np.zeros_like(total_delta)
    for window in windows:
        lead = int((archive.dates[window[-1]] - archive.dates[window[-2]]).astype(int))
        change = (archive.maps[window[-1]] - archive.maps[window[-2]])[core]
        valid = np.isfinite(change)
        total_delta[valid] += change[valid]
        total_days[valid] += lead
    if total_days.sum() == 0:
        raise ValueError("Training requires observed origin and target depths.")
    global_rate = total_delta.sum() / total_days.sum()
    prior_days = 20 * REFERENCE_DAYS
    baseline = (total_delta + prior_days * global_rate) / (total_days + prior_days) * REFERENCE_DAYS
    return core, baseline


def _window_rows(model, archive, drivers, windows, sample=False):
    """Targets/origin gaps are omitted; older predictors remain NaN until imputed."""
    raws, changes, curvatures, horizons, baselines, counts = [], [], [], [], [], []
    rng = np.random.default_rng(42)
    core = model["core"]
    for window in windows:
        lead = int((archive.dates[window[-1]] - archive.dates[window[-2]]).astype(int))
        if not 0 < lead <= MAX_LEAD_DAYS:
            raise ValueError("Physics training/evaluation supports horizons of 1–40 days.")
        change = (archive.maps[window[-1]] - archive.maps[window[-2]])[core]
        valid = np.flatnonzero(np.isfinite(change))
        if sample and len(valid):
            valid = np.sort(rng.choice(valid, min(TRAIN_CELLS_PER_WINDOW, len(valid)), replace=False))
        if not len(valid):
            raise ValueError("Every window must contain observed origin and target depths.")
        raw, curvature, _ = raw_features(archive, core, model["baseline"], window[:-1], drivers)
        raws.append(raw[valid])
        changes.append(change[valid])
        curvatures.append(curvature[valid])
        horizons.append(np.full(len(valid), lead))
        baselines.append(model["baseline"][valid] * lead / REFERENCE_DAYS)
        counts.append(len(valid))
    return {
        "raw": np.concatenate(raws), "change": np.concatenate(changes),
        "curvature": np.concatenate(curvatures), "lead": np.concatenate(horizons),
        "baseline": np.concatenate(baselines), "counts": np.asarray(counts),
    }


def _score(rows, prediction):
    error = (prediction - rows["change"]) / FEET_TO_METRES
    starts = np.r_[0, np.cumsum(rows["counts"])[:-1]]
    window_mae = np.add.reduceat(np.abs(error), starts) / rows["counts"]
    persistence = np.add.reduceat(np.abs(rows["change"]) / FEET_TO_METRES, starts) / rows["counts"]
    return {
        "mae_ft": float(window_mae.mean()), "persistence_mae_ft": float(persistence.mean()),
        "pooled_mae_ft": float(np.abs(error).mean()), "rmse_ft": float(np.sqrt(np.mean(error**2))),
        "bias_ft": float(error.mean()), "windows": len(rows["counts"]), "scored_cells": len(error),
    }


def _predict_rows(model, features, curvature, lead, baseline):
    """Portable float32 forward pass matching the training network equation."""
    output = []
    with threadpool_limits(limits=4):
        for member in model["members"]:
            parts = []
            for begin in range(0, len(features), 32768):
                sl = slice(begin, begin + 32768)
                value = features[sl]
                for index, (weight, bias) in enumerate(member["layers"]):
                    value = value @ weight.T + bias
                    if index < 2:
                        value = np.tanh(value)
                days = np.asarray(lead[sl], dtype=np.float32)
                source = value[:, 0] * np.float32(model["target_scale_m"]) * days / REFERENCE_DAYS
                transport = np.float32(member["kappa_m2_day"]) * days * np.asarray(curvature[sl], dtype=np.float32)
                parts.append(np.asarray(baseline[sl], dtype=np.float32) + source + transport)
            output.append(np.concatenate(parts))
    return np.mean(output, axis=0)


def train_model(archive, raw_dataset_path, windows, split):
    """Fit our own two-seed source/diffusion ensemble from the pinned v2 data.

    Validation selects each seed's checkpoint; test is read only after both fits.
    Median imputation does not fill maps or labels.  It only supplies older-map
    and driver predictors, with explicit missingness flags.  Recent measured
    changes, including dredging, remain the primary forecast target.
    """
    import torch
    from torch import nn

    windows = np.asarray(windows, dtype=int)
    split = np.asarray(split)
    if windows.ndim != 2 or windows.shape[1] != 5 or split.shape != (len(windows),):
        raise ValueError("Expected aligned five-map windows and train/val/test split labels.")
    if (np.diff(windows, axis=1) <= 0).any() or not set(split).issubset({"train", "val", "test"}):
        raise ValueError("Windows must be chronological and split labels must be train/val/test.")
    if any(not (split == label).any() for label in ("train", "val", "test")):
        raise ValueError("Separate train, validation and test windows are required.")
    if not (
        archive.dates[windows[split == "train", -1]].max() < archive.dates[windows[split == "val", 0]].min()
        and archive.dates[windows[split == "val", -1]].max() < archive.dates[windows[split == "test", 0]].min()
    ):
        raise ValueError("Training, validation and test survey dates must be disjoint and chronological.")
    if archive.grid["cell_metres"] != 10:
        raise ValueError("The fitted diffusion bound and benchmark require a 10 m grid.")
    torch.set_num_threads(4)
    start = time.perf_counter()
    drivers = read_drivers(raw_dataset_path, archive.dates)
    training = windows[split == "train"]
    core, baseline = _calibration(archive, training)
    model = {
        "version": 1, "sklearn": sklearn.__version__, "source_commit": SOURCE_COMMIT,
        "architecture": "physics-diffusion", "profile_id": "physics-diffusion",
        "model_name": "Physics-informed source + diffusion", "spatial": False,
        "core": core, "grid": dict(archive.grid), "baseline": baseline,
        "train_end": str(archive.dates[training[:, -1]].max()),
        "max_history_gap": int(np.diff(archive.dates[training[:, :-1]], axis=1).astype(int).max()),
        "target_months": sorted(set((archive.dates[windows[split != "train", -1]].astype("datetime64[M]").astype(int) % 12 + 1).tolist())),
        "input_features": list(FEATURE_NAMES) + [f"{name}_missing" for name in FEATURE_NAMES],
        "base_features": list(FEATURE_NAMES), "driver_names": list(DRIVERS),
        "driver_windows_days": [7, 28], "driver_min_coverage": 0.8,
        "future_drivers_used": False, "members": [], "training_history": [],
        "physics": {
            "equation": "delta_m=(T/28)*(baseline28_m+target_scale_m*f_theta(X))+kappa*T*L(depth_m)",
            "kappa_bounds_m2_day": [0.0, KAPPA_MAX], "maximum_lead_days": MAX_LEAD_DAYS,
            "boundary": "No flux across missing/non-core graph edges; equal and opposite observed-neighbour exchange.",
            "limitation": "Reduced source/diffusion surrogate; no identified sediment flux, boundary currents, or dredging forcing. The learned source is unconstrained.",
        },
        "training_parameters": {
            "seeds": list(SEEDS), "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "sample_cells_per_window": TRAIN_CELLS_PER_WINDOW, "sampling_seed": 42,
            "optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0001,
            "gradient_clip_norm": 5, "source_network": [84, 32, 32, 1],
            "activation": "tanh", "selection": "Mean per-window validation MAE; checkpoints 0,1,3,...,36",
        },
    }
    train = _window_rows(model, archive, drivers, training, sample=True)
    validation = _window_rows(model, archive, drivers, windows[split == "val"])
    model["imputer"] = SimpleImputer(strategy="median", keep_empty_features=True).fit(train["raw"])
    model["scaler"] = StandardScaler().fit(model["imputer"].transform(train["raw"]))
    model["training_missing_fraction"] = np.isnan(train["raw"]).mean(axis=0)
    model["training_rows"] = len(train["raw"])
    model["training_windows"] = len(training)
    model["target_scale_m"] = float(np.std(train["change"] - train["baseline"]))
    if model["target_scale_m"] <= 0:
        raise ValueError("Training requires nonzero target residual variation.")
    train_x = design_matrix(model, train.pop("raw"))
    val_x = design_matrix(model, validation.pop("raw"))
    target_scale = model["target_scale_m"]

    class SourceDiffusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.source = nn.Sequential(nn.Linear(84, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
            nn.init.zeros_(self.source[-1].weight)
            nn.init.zeros_(self.source[-1].bias)
            self.kappa = nn.Parameter(torch.tensor(0.01))

        def forward(self, x, days, lap, base):
            return base + self.source(x).squeeze(1) * target_scale * days / REFERENCE_DAYS + self.kappa * days * lap

    tx = torch.from_numpy(train_x)
    tlead = torch.tensor(train["lead"], dtype=torch.float32)
    tlap = torch.tensor(train["curvature"], dtype=torch.float32)
    tbase = torch.tensor(train["baseline"], dtype=torch.float32)
    ty = torch.tensor(train["change"], dtype=torch.float32)

    def validation_predictions(network):
        network.eval()
        values = []
        with torch.no_grad():
            for begin in range(0, len(val_x), 32768):
                sl = slice(begin, begin + 32768)
                prediction = network(
                    torch.from_numpy(val_x[sl]), torch.tensor(validation["lead"][sl], dtype=torch.float32),
                    torch.tensor(validation["curvature"][sl], dtype=torch.float32),
                    torch.tensor(validation["baseline"][sl], dtype=torch.float32),
                )
                values.append(prediction.numpy())
        return np.concatenate(values)

    for seed in SEEDS:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        network = SourceDiffusion()
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=0.0001)
        best = _score(validation, validation_predictions(network))["mae_ft"]
        best_state, best_epoch = copy.deepcopy(network.state_dict()), 0
        for epoch in range(1, EPOCHS + 1):
            network.train()
            order = rng.permutation(len(tx))
            losses = []
            for begin in range(0, len(order), BATCH_SIZE):
                indices = order[begin:begin + BATCH_SIZE]
                prediction = network(tx[indices], tlead[indices], tlap[indices], tbase[indices])
                loss = ((prediction - ty[indices]) / target_scale).square().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 5)
                optimizer.step()
                with torch.no_grad():
                    network.kappa.clamp_(0, KAPPA_MAX)
                losses.append(loss.item())
            if epoch == 1 or epoch % 3 == 0:
                score = _score(validation, validation_predictions(network))["mae_ft"]
                model["training_history"].append({
                    "seed": seed, "epoch": epoch, "training_loss": float(np.mean(losses)),
                    "validation_mae_ft": score, "kappa_m2_day": float(network.kappa.detach()),
                })
                if score < best:
                    best, best_state, best_epoch = score, copy.deepcopy(network.state_dict()), epoch
        network.load_state_dict(best_state)
        layers = [(network.source[index].weight.detach().numpy().copy(), network.source[index].bias.detach().numpy().copy()) for index in (0, 2, 4)]
        model["members"].append({
            "seed": seed, "layers": layers, "kappa_m2_day": float(network.kappa.detach()),
            "best_epoch": best_epoch, "validation_mae_ft": best,
        })
        print(f"Physics seed {seed}: selected epoch {best_epoch}, validation MAE {best:.6f} ft, kappa {float(network.kappa.detach()):.6f} m2/day", flush=True)

    val_prediction = _predict_rows(model, val_x, validation["curvature"], validation["lead"], validation["baseline"])
    test = _window_rows(model, archive, drivers, windows[split == "test"])
    test_x = design_matrix(model, test.pop("raw"))
    test_prediction = _predict_rows(model, test_x, test["curvature"], test["lead"], test["baseline"])
    model["evaluation"] = {"validation": _score(validation, val_prediction), "test": _score(test, test_prediction)}
    model["fit_seconds"] = time.perf_counter() - start
    return model


def predict_change(model, archive, raw_dataset_path, history, lead=REFERENCE_DAYS):
    """Return one metre-valued change per core cell; absent origins remain NaN."""
    if not np.isscalar(lead) or not np.isfinite(lead) or not 0 < lead <= MAX_LEAD_DAYS:
        raise ValueError("Physics predictions support horizons of 1–40 days.")
    if archive.grid != model["grid"] or archive.maps.shape[1:] != model["core"].shape:
        raise ValueError("Align incoming maps to the training grid before physics prediction.")
    drivers = read_drivers(raw_dataset_path, archive.dates)
    raw, curvature, observed = raw_features(archive, model["core"], model["baseline"], history, drivers)
    result = np.full(int(model["core"].sum()), np.nan)
    if observed.any():
        days = np.full(int(observed.sum()), lead)
        result[observed] = _predict_rows(
            model, design_matrix(model, raw[observed]), curvature[observed], days,
            model["baseline"][observed] * lead / REFERENCE_DAYS,
        )
    return result
