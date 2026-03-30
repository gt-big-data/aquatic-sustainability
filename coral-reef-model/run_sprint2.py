"""Sprint 2 pipeline: baseline model, hyperparameter search, evaluation on GBR test set."""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import load_and_split, scale_features, train_baseline, MODELS_DIR
from src.tune import run_hyperparameter_search
from src.utils import TARGET_COL, SECTOR_COL, LAT_COL, LON_COL, YEAR_COL, PROJECT_ROOT

REPORTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'reports')


def run_sprint2():
    print("=" * 60)
    print("SPRINT 2: Baseline Model, Hyperparameter Tuning, Evaluation")
    print("=" * 60)

    # Step 1: Train/test split
    train_df, test_df, X_train, y_train, X_test, y_test, feature_cols = load_and_split()

    # Step 2: Feature scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, feature_cols)

    # Step 3: Baseline model
    baseline, baseline_cv_mae, baseline_cv_std, y_pred_baseline, naive_mae = train_baseline(
        X_train_scaled, y_train, X_test_scaled, y_test
    )

    # Step 4: Hyperparameter search
    search, tuned_cv_mae = run_hyperparameter_search(X_train_scaled, y_train)

    print(f"\nImprovement over baseline CV: {baseline_cv_mae - tuned_cv_mae:.4f}")

    # Step 5: Evaluate best model on GBR test set
    best_model = search.best_estimator_
    y_pred_tuned = np.clip(best_model.predict(X_test_scaled), 0.0, 1.0)

    y_test_100 = y_test * 100
    y_pred_tuned_100 = y_pred_tuned * 100
    y_pred_baseline_100 = y_pred_baseline * 100

    tuned_mae = mean_absolute_error(y_test_100, y_pred_tuned_100)
    tuned_rmse = np.sqrt(mean_squared_error(y_test_100, y_pred_tuned_100))
    tuned_r2 = r2_score(y_test_100, y_pred_tuned_100)

    baseline_test_mae = mean_absolute_error(y_test_100, y_pred_baseline_100)

    print(f"\n--- Tuned Model GBR Test Results ---")
    print(f"Tuned GBR Test MAE:  {tuned_mae:.2f}%")
    print(f"Tuned GBR Test RMSE: {tuned_rmse:.2f}%")
    print(f"Tuned GBR Test R2:   {tuned_r2:.4f}")
    print(f"Improvement over baseline: {baseline_test_mae - tuned_mae:.2f}%")
    print(f"Improvement over naive:    {naive_mae - tuned_mae:.2f}%")

    # Step 6: Save all artifacts
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Save best model
    best_model_path = os.path.join(MODELS_DIR, 'best_model.json')
    best_model.save_model(best_model_path)
    print(f"\nBest model saved: {best_model_path}")

    # Save CV results
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results_path = os.path.join(REPORTS_DIR, 'cv_results.csv')
    cv_results.to_csv(cv_results_path, index=False)
    print(f"CV results saved: {cv_results_path}")

    # Save test predictions with residuals
    test_results = test_df[[YEAR_COL, LAT_COL, LON_COL, SECTOR_COL, TARGET_COL]].copy()
    test_results['Predicted'] = y_pred_tuned
    test_results['Predicted_Pct'] = y_pred_tuned_100
    test_results['Actual_Pct'] = y_test_100.values
    test_results['Residual'] = y_pred_tuned_100 - y_test_100.values
    predictions_path = os.path.join(REPORTS_DIR, 'test_predictions.csv')
    test_results.to_csv(predictions_path, index=False)
    print(f"Test predictions saved: {predictions_path}")

    # Verify scaler exists
    scaler_path = os.path.join(PROJECT_ROOT, 'outputs', 'scalers', 'robust_scaler.pkl')
    assert os.path.exists(scaler_path), f"Scaler not found at {scaler_path}"

    # Step 7: Summary report
    print(f"\n{'=' * 60}")
    print(f"SPRINT 2 SUMMARY")
    print(f"{'=' * 60}")
    print(f"Training data: {len(X_train)} rows (global, pre-2016)")
    print(f"Test data:     {len(X_test)} rows (GBR, 2016-2017)")
    print(f"Features:      {len(feature_cols)}")
    print(f"\nCV Results (5-fold, global training set):")
    print(f"  Baseline MAE: {baseline_cv_mae:.4f}")
    print(f"  Tuned MAE:    {tuned_cv_mae:.4f}")
    print(f"\nGBR Test Set Results (0-100% scale):")
    print(f"  Naive MAE:    {naive_mae:.2f}%")
    print(f"  Baseline MAE: {baseline_test_mae:.2f}%")
    print(f"  Tuned MAE:    {tuned_mae:.2f}%")
    print(f"  Tuned RMSE:   {tuned_rmse:.2f}%")
    print(f"  Tuned R2:     {tuned_r2:.4f}")
    print(f"\nBest Hyperparameters:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    print(f"\nArtifacts saved:")
    print(f"  - outputs/models/best_model.json")
    print(f"  - outputs/models/baseline_model.json")
    print(f"  - outputs/scalers/robust_scaler.pkl")
    print(f"  - outputs/reports/cv_results.csv")
    print(f"  - outputs/reports/test_predictions.csv")

    # Per-sector breakdown
    print(f"\nPer-Sector GBR Test Breakdown:")
    for sector in ['Northern', 'Central', 'Southern']:
        mask = test_results[SECTOR_COL] == sector
        if mask.sum() > 0:
            sector_mae = mean_absolute_error(
                test_results.loc[mask, 'Actual_Pct'],
                test_results.loc[mask, 'Predicted_Pct'],
            )
            sector_mean_actual = test_results.loc[mask, 'Actual_Pct'].mean()
            print(f"  {sector}: MAE={sector_mae:.2f}%, n={mask.sum()}, mean_actual={sector_mean_actual:.1f}%")

    print(f"{'=' * 60}")


if __name__ == '__main__':
    run_sprint2()
