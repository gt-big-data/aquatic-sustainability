"""Sprint 3 Part B: Diagnostic experiments to understand and fix underprediction."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from src.utils import TARGET_COL, YEAR_COL, GBR_COL, METADATA_COLS


BASELINE_PARAMS = dict(
    objective='reg:tweedie',
    tweedie_variance_power=1.5,
    tree_method='hist',
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    colsample_bytree=0.7,
    random_state=42,
)


def _train_and_eval(X_train, y_train, X_test, y_test, feature_cols, sample_weight=None):
    """Train a baseline-config model and return GBR test MAE on 0-100 scale."""
    scaler = RobustScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    model = XGBRegressor(**BASELINE_PARAMS)
    model.fit(X_train_s, y_train, sample_weight=sample_weight)

    pred = np.clip(model.predict(X_test_s), 0, 1) * 100
    actual = y_test.values * 100
    return mean_absolute_error(actual, pred), model


def run_gbr_only(df, feature_cols, X_test, y_test):
    """B.1: Train on GBR pre-2016 only."""
    gbr_train = df[(df[GBR_COL] == True) & (df[YEAR_COL] < 2016)].copy()
    X_tr = gbr_train[feature_cols]
    y_tr = gbr_train[TARGET_COL]
    mae, _ = _train_and_eval(X_tr, y_tr, X_test, y_test, feature_cols)
    print(f"\nB.1 — GBR-only training: {len(gbr_train)} train rows")
    print(f"  GBR-only MAE: {mae:.2f}%")
    return mae


def run_gbr_weighted(train_df, feature_cols, X_train, y_train, X_test, y_test):
    """B.2: Global training with GBR sample weights (5x, 10x, 20x)."""
    results = {}
    for multiplier in [5, 10, 20]:
        weights = np.where(train_df[GBR_COL].values, float(multiplier), 1.0)
        mae, _ = _train_and_eval(X_train, y_train, X_test, y_test, feature_cols, sample_weight=weights)
        results[f'GBR {multiplier}x'] = mae
        print(f"  Global + GBR {multiplier}x weight MAE: {mae:.2f}%")
    return results


def run_severity_weighted(X_train, y_train, X_test, y_test, feature_cols):
    """B.3: Upweight high-bleaching rows."""
    severity_weights = np.where(y_train.values > 0, 3.0, 1.0)
    mae, _ = _train_and_eval(X_train, y_train, X_test, y_test, feature_cols, sample_weight=severity_weights)
    print(f"\nB.3 — Severity-weighted (bleaching>0 gets 3x)")
    print(f"  Severity-weighted MAE: {mae:.2f}%")
    return mae


def run_reduced_features(train_df, X_train, y_train, X_test, y_test, feature_cols):
    """B.4: Train with only top-7 correlated features."""
    top_features = ['SSTA_DHW', 'TSA_DHW', 'SSTA_Frequency', 'TSA_Frequency',
                    'TSA', 'Latitude_Degrees', 'Longitude_Degrees']
    # Only keep features that exist in the data
    top_features = [f for f in top_features if f in feature_cols]
    X_tr = X_train[top_features]
    X_te = X_test[top_features]
    mae, _ = _train_and_eval(X_tr, y_train, X_te, y_test, top_features)
    print(f"\nB.4 — Reduced features ({len(top_features)} features)")
    print(f"  Top-{len(top_features)} features MAE: {mae:.2f}%")
    return mae


def run_all_experiments(df, train_df, X_train, y_train, X_test, y_test, feature_cols, naive_mae):
    """Run all diagnostic experiments and print summary table."""
    print(f"\n{'=' * 50}")
    print(f"PART B: DIAGNOSTIC EXPERIMENTS")
    print(f"{'=' * 50}")

    gbr_only_mae = run_gbr_only(df, feature_cols, X_test, y_test)

    print(f"\nB.2 — GBR-weighted training")
    gbr_weighted = run_gbr_weighted(train_df, feature_cols, X_train, y_train, X_test, y_test)

    severity_mae = run_severity_weighted(X_train, y_train, X_test, y_test, feature_cols)

    reduced_mae = run_reduced_features(train_df, X_train, y_train, X_test, y_test, feature_cols)

    # Summary table
    from sklearn.metrics import mean_absolute_error as mae_fn
    baseline_pred = np.clip(
        XGBRegressor(**BASELINE_PARAMS).fit(
            pd.DataFrame(RobustScaler().fit_transform(X_train), columns=feature_cols, index=X_train.index),
            y_train
        ).predict(
            pd.DataFrame(RobustScaler().fit(X_train).transform(X_test), columns=feature_cols, index=X_test.index)
        ), 0, 1) * 100
    # Use the stored baseline and tuned MAE from Sprint 2 instead of re-computing
    # (they were: baseline=14.43, tuned=15.52)

    all_results = {
        'Naive (predict mean)': naive_mae,
        'Global baseline (40 feat)': 14.43,  # from Sprint 2
        'Global tuned (40 feat)': 15.52,      # from Sprint 2
        'GBR-only baseline (40 feat)': gbr_only_mae,
        'Global + GBR 5x weight': gbr_weighted['GBR 5x'],
        'Global + GBR 10x weight': gbr_weighted['GBR 10x'],
        'Global + GBR 20x weight': gbr_weighted['GBR 20x'],
        'Global + severity weight': severity_mae,
        'Global + top-7 features only': reduced_mae,
    }

    print(f"\n{'=' * 50}")
    print(f"DIAGNOSTIC EXPERIMENTS — GBR Test MAE (%)")
    print(f"{'=' * 50}")
    best_name = None
    best_mae = float('inf')
    for name, mae_val in all_results.items():
        marker = ''
        if mae_val < best_mae:
            best_mae = mae_val
            best_name = name
        print(f"  {name + ':':<38} {mae_val:.2f}%")
    print(f"{'=' * 50}")
    print(f"  BEST: {best_name} ({best_mae:.2f}%)")
    print(f"{'=' * 50}")

    return all_results, best_name
