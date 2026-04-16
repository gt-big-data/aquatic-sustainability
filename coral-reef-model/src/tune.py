"""Sprint 2: Hyperparameter search via RandomizedSearchCV."""
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold
from xgboost import XGBRegressor


PARAM_GRID = {
    'tweedie_variance_power': [1.2, 1.3, 1.4, 1.5, 1.6],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
    'n_estimators': [300, 500, 800],
    'min_child_weight': [3, 5, 10],
}


def run_hyperparameter_search(X_train_scaled, y_train):
    """Run 50-combo RandomizedSearchCV with 5-fold KFold."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        XGBRegressor(
            objective='reg:tweedie',
            tree_method='hist',
            random_state=42,
        ),
        param_distributions=PARAM_GRID,
        n_iter=50,
        cv=cv,
        scoring='neg_mean_absolute_error',
        random_state=42,
        verbose=1,
        n_jobs=-1,
        refit=True,
    )

    print(f"\n--- Hyperparameter Search ---")
    print(f"50 iterations x 5 folds = 250 model fits")
    print(f"Training rows: {len(X_train_scaled)}, features: {X_train_scaled.shape[1]}")
    print(f"Starting search...\n")

    search.fit(X_train_scaled, y_train)

    best_cv_mae = -search.best_score_
    print(f"\nBest CV MAE: {best_cv_mae:.4f}")
    print(f"Best params: {search.best_params_}")

    return search, best_cv_mae
