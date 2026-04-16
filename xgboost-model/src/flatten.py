"""
Flatten 3D sequence data into tabular features for XGBoost.

Input:  sequences_reduced_16.npz — X (N, 16, 4), y (N,), meta (N, 4)
Output: Tabular DataFrame with 36 features (4 satellite × 8 stats + 4 metadata)
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_and_flatten(data_path: str = "data/sequences_reduced_16.npz") -> tuple:
    """Load sequence data and flatten to tabular features.

    For each of the 4 satellite features across the 16-week window, compute:
    mean, max, min, std, last, first, trend (last-first), slope (linear).
    Also adds latitude, longitude, year, month from metadata.

    Returns:
        X_flat: pd.DataFrame of shape (N, 36)
        y: np.ndarray of shape (N,)
        meta: np.ndarray of shape (N, 4)
    """
    data_path = Path(data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = np.load(data_path, allow_pickle=True)
    X_seq = data["X"]  # (28539, 16, 4)
    y = data["y"]  # (28539,)
    meta = data["meta"]  # (28539, 4) — lat, lon, year, month
    feature_names = list(data["feature_names"])

    print(f"Loaded: {X_seq.shape[0]} samples, {X_seq.shape[1]} weeks, {X_seq.shape[2]} features")
    print(f"Classes: {np.bincount(y.astype(int))} (none/moderate/severe)")

    # Summary statistics per feature across the 16-week window
    flat_features = {}

    for i, fname in enumerate(feature_names):
        series = X_seq[:, :, i]  # (N, 16)

        flat_features[f"{fname}_mean"] = np.nanmean(series, axis=1)
        flat_features[f"{fname}_max"] = np.nanmax(series, axis=1)
        flat_features[f"{fname}_min"] = np.nanmin(series, axis=1)
        flat_features[f"{fname}_std"] = np.nanstd(series, axis=1)
        flat_features[f"{fname}_last"] = series[:, -1]
        flat_features[f"{fname}_first"] = series[:, 0]
        flat_features[f"{fname}_trend"] = series[:, -1] - series[:, 0]

        # Vectorized linear slope over the 16 weeks
        x_time = np.arange(series.shape[1])
        x_mean = x_time.mean()
        x_var = ((x_time - x_mean) ** 2).sum()
        slopes = (
            (series - series.mean(axis=1, keepdims=True)) * (x_time - x_mean)
        ).sum(axis=1) / x_var
        flat_features[f"{fname}_slope"] = slopes

    # Static features from metadata
    flat_features["latitude"] = meta[:, 0]
    flat_features["longitude"] = meta[:, 1]
    flat_features["year"] = meta[:, 2]
    flat_features["month"] = meta[:, 3]

    X_flat = pd.DataFrame(flat_features)

    nan_count = X_flat.isna().sum().sum()
    print(f"Flattened shape: {X_flat.shape}")
    print(f"NaN check: {nan_count} total NaNs")

    X_flat = X_flat.fillna(0)

    return X_flat, y, meta


if __name__ == "__main__":
    X_flat, y, meta = load_and_flatten()
    print(f"\nFeature columns ({len(X_flat.columns)}):")
    for col in X_flat.columns:
        print(f"  {col}")
