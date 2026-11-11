"""Model profiles fixed before scoring the 2025–2026 holdout."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    label: str
    family: str
    spatial: bool


PROFILES = {
    "extra-trees": Profile("Extra Trees (incumbent)", "Extra Trees", False),
    "extra-trees-spatial": Profile("Extra Trees + neighborhoods", "Extra Trees", True),
    "hist-boosting": Profile("Histogram gradient boosting", "Histogram gradient boosting", True),
    "xgb-survey": Profile("XGBoost (survey features)", "XGBoost", False),
    "xgb-spatial": Profile("XGBoost + neighborhoods", "XGBoost", True),
    "lightgbm-31": Profile("LightGBM (31 leaves)", "LightGBM", True),
    "lightgbm-95": Profile("LightGBM (95 leaves)", "LightGBM", True),
    "catboost-rmse": Profile("CatBoost (squared loss)", "CatBoost", True),
    "catboost-mae": Profile("CatBoost (absolute loss)", "CatBoost", True),
}


def fit_estimator(profile_id, features, changes, weights, validation):
    profile = PROFILES[profile_id]
    val_features, val_changes, val_weights = validation
    if profile.family == "Extra Trees":
        from sklearn.ensemble import ExtraTreesRegressor

        estimator = ExtraTreesRegressor(
            n_estimators=120, max_depth=18, min_samples_leaf=20, max_features=0.8,
            n_jobs=4, random_state=42,
        )
        return estimator.fit(features, changes, sample_weight=weights)

    if profile.family == "Histogram gradient boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        estimator = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.06, max_iter=400, max_leaf_nodes=63,
            min_samples_leaf=70, l2_regularization=3, early_stopping=False, random_state=42,
        )
        return estimator.fit(features, changes, sample_weight=weights)

    if profile.family == "XGBoost":
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            objective="reg:pseudohubererror", huber_slope=0.25, eval_metric="mae",
            tree_method="hist", n_estimators=900, learning_rate=0.045, max_depth=7,
            min_child_weight=35, subsample=0.85, colsample_bytree=0.9, reg_lambda=8,
            early_stopping_rounds=45, n_jobs=4, random_state=42,
        )
        return estimator.fit(
            features, changes, sample_weight=weights, eval_set=[(val_features, val_changes)],
            sample_weight_eval_set=[val_weights], verbose=False,
        )

    if profile.family == "LightGBM":
        from lightgbm import LGBMRegressor, early_stopping

        leaves = 31 if profile_id == "lightgbm-31" else 95
        estimator = LGBMRegressor(
            objective="regression_l1", n_estimators=900, learning_rate=0.045, num_leaves=leaves,
            min_child_samples=100, reg_lambda=5, colsample_bytree=0.9, subsample=0.85,
            subsample_freq=1, n_jobs=4, random_state=42, verbosity=-1,
            deterministic=True, force_col_wise=True,
        )
        return estimator.fit(
            features, changes, sample_weight=weights, eval_set=[(val_features, val_changes)],
            eval_sample_weight=[val_weights], eval_metric="l1", callbacks=[early_stopping(45, verbose=False)],
        )

    from catboost import CatBoostRegressor, Pool

    estimator = CatBoostRegressor(
        loss_function="RMSE" if profile_id == "catboost-rmse" else "MAE",
        eval_metric="MAE", depth=7, iterations=1000, learning_rate=0.06, l2_leaf_reg=6,
        early_stopping_rounds=45, thread_count=4, random_seed=42, allow_writing_files=False,
        verbose=False,
    )
    return estimator.fit(
        Pool(features, changes, weight=weights),
        eval_set=Pool(val_features, val_changes, weight=val_weights), use_best_model=True,
    )


def predict(estimator, features):
    # The native booster accepts NumPy arrays without sklearn's feature-name warning.
    booster = getattr(estimator, "booster_", None)
    return booster.predict(features, num_threads=4) if booster is not None else estimator.predict(features)
