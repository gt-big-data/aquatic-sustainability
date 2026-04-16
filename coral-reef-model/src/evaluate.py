"""Sprint 3 Part A: Full evaluation and visualization of baseline and tuned models."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from xgboost import XGBRegressor

from src.utils import TARGET_COL, SECTOR_COL, YEAR_COL, GBR_COL, METADATA_COLS, FIGURES_DIR


def load_artifacts():
    """Load both models, scaler, and re-split data."""
    import joblib
    from src.utils import PROCESSED_PATH, PROJECT_ROOT

    baseline = XGBRegressor()
    baseline.load_model(os.path.join(PROJECT_ROOT, 'outputs', 'models', 'baseline_model.json'))

    tuned = XGBRegressor()
    tuned.load_model(os.path.join(PROJECT_ROOT, 'outputs', 'models', 'best_model.json'))

    scaler = joblib.load(os.path.join(PROJECT_ROOT, 'outputs', 'scalers', 'robust_scaler.pkl'))

    df = pd.read_csv(PROCESSED_PATH)
    train_df = df[df[YEAR_COL] < 2016].copy()
    test_df = df[(df[GBR_COL] == True) & (df[YEAR_COL] >= 2016)].copy()

    feature_cols = [c for c in df.columns if c not in [TARGET_COL] + METADATA_COLS]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    return (baseline, tuned, scaler, train_df, test_df,
            X_train_scaled, y_train, X_test_scaled, y_test, feature_cols)


def model_comparison_table(baseline, tuned, X_test_scaled, y_test, y_train):
    """A.2: Side-by-side model comparison including severe-bleaching-only MAE."""
    y_test_100 = y_test.values * 100

    baseline_pred_100 = np.clip(baseline.predict(X_test_scaled), 0, 1) * 100
    tuned_pred_100 = np.clip(tuned.predict(X_test_scaled), 0, 1) * 100
    naive_pred_100 = np.full_like(y_test_100, y_train.mean() * 100)

    results = {}
    for name, pred in [('Naive', naive_pred_100), ('Baseline', baseline_pred_100), ('Tuned', tuned_pred_100)]:
        results[name] = {
            'MAE (%)': mean_absolute_error(y_test_100, pred),
            'RMSE (%)': np.sqrt(mean_squared_error(y_test_100, pred)),
            'R²': r2_score(y_test_100, pred),
            'Median Abs Error (%)': median_absolute_error(y_test_100, pred),
        }

    print(f"\n{'=' * 60}")
    print(f"SIDE-BY-SIDE MODEL COMPARISON — GBR Test Set (n={len(y_test)})")
    print(f"{'=' * 60}")
    print(f"{'Metric':<22} {'Naive':>10} {'Baseline':>10} {'Tuned':>10}")
    print(f"{'-' * 52}")
    for metric in ['MAE (%)', 'RMSE (%)', 'R²', 'Median Abs Error (%)']:
        fmt = '.2f' if metric != 'R²' else '.4f'
        print(f"{metric:<22} {results['Naive'][metric]:>10{fmt}} {results['Baseline'][metric]:>10{fmt}} {results['Tuned'][metric]:>10{fmt}}")

    # Severe bleaching analysis (>20% actual)
    severe_mask = y_test_100 > 20
    print(f"\n--- Severe Bleaching Only (actual > 20%): n={severe_mask.sum()} ---")
    for name, pred in [('Naive', naive_pred_100), ('Baseline', baseline_pred_100), ('Tuned', tuned_pred_100)]:
        mae = mean_absolute_error(y_test_100[severe_mask], pred[severe_mask])
        print(f"  {name} MAE on severe: {mae:.2f}%")

    return results, baseline_pred_100, tuned_pred_100, naive_pred_100


def plot_feature_importance(baseline, feature_cols):
    """A.3: Gain-based feature importance bar plot."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    importance = baseline.get_booster().get_score(importance_type='gain')
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([x[0] for x in sorted_imp], [x[1] for x in sorted_imp])
    ax.set_xlabel('Gain')
    ax.set_title('Top 15 Feature Importance (Gain) — Baseline Model')
    ax.invert_yaxis()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'feature_importance_gain.png')
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"\nFeature importance plot saved: {path}")

    print("\nTop 15 features by gain:")
    for name, gain in sorted_imp:
        print(f"  {name}: {gain:.1f}")

    return sorted_imp


def plot_shap_summary(baseline, X_test_scaled):
    """A.3: SHAP summary plot."""
    try:
        import shap
    except ImportError:
        print("\nSHAP not installed — skipping SHAP summary plot.")
        print("Install with: pip3 install --break-system-packages shap")
        return None

    os.makedirs(FIGURES_DIR, exist_ok=True)

    explainer = shap.TreeExplainer(baseline)
    shap_values = explainer.shap_values(X_test_scaled)

    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_scaled, show=False, max_display=15)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'shap_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved: {path}")

    return shap_values


def plot_residual_analysis(baseline, X_test_scaled, y_test, test_df):
    """A.4: 4-panel residual analysis."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    baseline_pred_100 = np.clip(baseline.predict(X_test_scaled), 0, 1) * 100
    y_test_100 = y_test.values * 100
    residuals = baseline_pred_100 - y_test_100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Residual distribution
    axes[0, 0].hist(residuals, bins=25, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Residual (Predicted - Actual) %')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Residual Distribution')
    skew_label = 'negative (underprediction)' if np.mean(residuals) < 0 else 'positive (overprediction)'
    axes[0, 0].annotate(f'Mean: {np.mean(residuals):.1f}% ({skew_label})',
                        xy=(0.05, 0.95), xycoords='axes fraction', va='top', fontsize=9)

    # 2. Predicted vs Actual
    axes[0, 1].scatter(y_test_100, baseline_pred_100, alpha=0.5, s=20)
    axes[0, 1].plot([0, 100], [0, 100], 'r--', label='Perfect prediction')
    axes[0, 1].set_xlabel('Actual Bleaching %')
    axes[0, 1].set_ylabel('Predicted Bleaching %')
    axes[0, 1].set_title('Predicted vs Actual')
    axes[0, 1].legend()

    # 3. Residuals by sector
    colors = {'Northern': '#e41a1c', 'Central': '#377eb8', 'Southern': '#4daf4a'}
    for sector in ['Northern', 'Central', 'Southern']:
        mask = test_df[SECTOR_COL].values == sector
        if mask.sum() > 0:
            axes[1, 0].scatter(y_test_100[mask], residuals[mask],
                               label=f'{sector} (n={mask.sum()})', alpha=0.5, s=20,
                               color=colors[sector])
    axes[1, 0].axhline(0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Actual Bleaching %')
    axes[1, 0].set_ylabel('Residual')
    axes[1, 0].set_title('Residuals by Sector')
    axes[1, 0].legend()

    # 4. Residuals by year
    for year in [2016, 2017]:
        mask = test_df[YEAR_COL].values == year
        if mask.sum() > 0:
            axes[1, 1].scatter(y_test_100[mask], residuals[mask],
                               label=f'{year} (n={mask.sum()})', alpha=0.5, s=20)
    axes[1, 1].axhline(0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Actual Bleaching %')
    axes[1, 1].set_ylabel('Residual')
    axes[1, 1].set_title('Residuals by Year')
    axes[1, 1].legend()

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'residual_analysis.png')
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"\nResidual analysis plot saved: {path}")

    return residuals


def plot_prediction_distribution(baseline, X_test_scaled, y_test):
    """A.5: Actual vs predicted distribution comparison."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    baseline_pred_100 = np.clip(baseline.predict(X_test_scaled), 0, 1) * 100
    y_test_100 = y_test.values * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(y_test_100, bins=20, alpha=0.5, label='Actual', edgecolor='black')
    ax.hist(baseline_pred_100, bins=20, alpha=0.5, label='Predicted', edgecolor='black')
    ax.set_xlabel('Bleaching %')
    ax.set_ylabel('Count')
    ax.set_title('Distribution: Actual vs Predicted Bleaching (GBR Test Set)')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'prediction_distribution.png')
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Prediction distribution plot saved: {path}")
