"""Sprint 2: Train/test split, feature scaling, baseline model training and evaluation."""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.utils import TARGET_COL, YEAR_COL, GBR_COL, SECTOR_COL, METADATA_COLS, PROCESSED_PATH, PROJECT_ROOT


SCALERS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'scalers')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'models')


def load_and_split():
    """Load processed data and split into train/test sets."""
    df = pd.read_csv(PROCESSED_PATH)
    print(f"Loaded {len(df)} rows from {PROCESSED_PATH}")

    # Train: all global data before 2016
    train_df = df[df[YEAR_COL] < 2016].copy()

    # Test: GBR only, 2016+
    test_df = df[(df[GBR_COL] == True) & (df[YEAR_COL] >= 2016)].copy()

    # Feature columns = everything except target + metadata
    feature_cols = [c for c in df.columns if c not in [TARGET_COL] + METADATA_COLS]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    # Verification
    print(f"\n--- Train/Test Split ---")
    print(f"Train: {len(train_df)} rows, years {int(train_df[YEAR_COL].min())}–{int(train_df[YEAR_COL].max())}")
    print(f"Test:  {len(test_df)} rows, years {int(test_df[YEAR_COL].min())}–{int(test_df[YEAR_COL].max())}")
    print(f"Features: {len(feature_cols)}")
    print(f"\nTrain target: mean={y_train.mean():.4f}, % zeros={( y_train == 0).mean() * 100:.1f}%")
    print(f"Test target:  mean={y_test.mean():.4f}, % zeros={(y_test == 0).mean() * 100:.1f}%")

    # Confirm no data leakage (no overlapping indices)
    assert len(set(train_df.index) & set(test_df.index)) == 0, "Data leakage: overlapping indices!"
    print(f"No data leakage confirmed (0 overlapping indices)")

    return train_df, test_df, X_train, y_train, X_test, y_test, feature_cols


def scale_features(X_train, X_test, feature_cols):
    """Fit RobustScaler on train, transform both train and test."""
    os.makedirs(SCALERS_DIR, exist_ok=True)

    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index,
    )

    scaler_path = os.path.join(SCALERS_DIR, 'robust_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"\n--- Feature Scaling ---")
    print(f"Scaler saved: {scaler_path}")

    # Sanity check: print center/scale for top-5 features
    print(f"\nScaler center (median) and scale (IQR) for top-5 features:")
    for i, col in enumerate(feature_cols[:5]):
        print(f"  {col}: center={scaler.center_[i]:.4f}, scale={scaler.scale_[i]:.4f}")

    return X_train_scaled, X_test_scaled, scaler


def train_baseline(X_train_scaled, y_train, X_test_scaled, y_test):
    """Train baseline XGBoost, compute CV and test metrics."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    baseline = XGBRegressor(
        objective='reg:tweedie',
        tweedie_variance_power=1.5,
        tree_method='hist',
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        colsample_bytree=0.7,
        random_state=42,
    )

    # 5-fold CV on training data
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        baseline, X_train_scaled, y_train,
        cv=cv,
        scoring='neg_mean_absolute_error',
    )
    baseline_cv_mae = -scores.mean()
    baseline_cv_std = scores.std()
    print(f"\n--- Baseline Model ---")
    print(f"CV MAE: {baseline_cv_mae:.4f} +/- {baseline_cv_std:.4f}")

    # Fit on full training data
    baseline.fit(X_train_scaled, y_train)
    y_pred_baseline = np.clip(baseline.predict(X_test_scaled), 0.0, 1.0)

    # Metrics on 0-100 scale
    y_test_100 = y_test * 100
    y_pred_100 = y_pred_baseline * 100

    baseline_test_mae = mean_absolute_error(y_test_100, y_pred_100)
    baseline_test_rmse = np.sqrt(mean_squared_error(y_test_100, y_pred_100))
    baseline_test_r2 = r2_score(y_test_100, y_pred_100)

    print(f"\nBaseline GBR Test MAE:  {baseline_test_mae:.2f}%")
    print(f"Baseline GBR Test RMSE: {baseline_test_rmse:.2f}%")
    print(f"Baseline GBR Test R2:   {baseline_test_r2:.4f}")

    # Naive baseline (predict training mean)
    naive_pred = np.full_like(y_test_100, y_train.mean() * 100)
    naive_mae = mean_absolute_error(y_test_100, naive_pred)
    print(f"\nNaive baseline MAE (predict training mean): {naive_mae:.2f}%")

    # Save baseline model
    baseline_path = os.path.join(MODELS_DIR, 'baseline_model.json')
    baseline.save_model(baseline_path)
    print(f"Baseline model saved: {baseline_path}")

    return baseline, baseline_cv_mae, baseline_cv_std, y_pred_baseline, naive_mae
