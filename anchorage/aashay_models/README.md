# Anchorage Harbor: 10 depth-change models

Same task, data and splits as `anchorage_convlstm_drivers.ipynb`: four depth maps in, change in each 10 m cell ~4 weeks after the last map out (`timeseries_v2`, train 2014–22, val 2023–24, test 2025–26). Scored with masked MAE in ft. Hot-spot MAE covers cells that filled more than 0.75 ft per 4 weeks on average in the training years. AUROC/AP rank cells that shoal by more than 1 ft. Every model was tuned on val only; test was not used for any selection.

```
pip install numpy scipy scikit-learn lightgbm torch pandas tabulate
cd anchorage/aashay_models
python m00_baselines.py && python m02_lightgbm.py && python m03_hurdle.py   # m03 reuses m02's best params
python m01_linear.py ... m10_timesformer.py && python summarize.py
```

| # | file | model | how it was tuned |
|---|------|-------|------------------|
| 1 | `m01_linear.py` | Spline GAM (ridge/Huber) on the residual over climatology | linear vs GAM, ridge vs Huber, alpha, picked on val |
| 2 | `m02_lightgbm.py` | LightGBM | 14-config random search (`gbm.py`), average of the top 3 at lr 0.03 |
| 3 | `m03_hurdle.py` | Hurdle: shoal/stable/deepen classifier × per-class regressors | m02's best params; threshold τ and combine rule picked on val |
| 4 | `m04_holt.py` | Damped-trend Holt per cell over the last 12 surveys + climatology | grid over α, β, φ and climatology/trend weights, fit on train |
| 5 | `m05_eof_rrr.py` | EOF (PCA) reduced-rank regression | leave-one-year-out CV within train; local relative depth |
| 6 | `m06_unet.py` | U-Net | shared deep setup below |
| 7 | `m07_convgru.py` | 2-layer ConvGRU | " |
| 8 | `m08_simvp.py` | SimVP | " |
| 9 | `m09_fno.py` | Fourier Neural Operator | " |
| 10 | `m10_timesformer.py` | TimeSformer-lite (divided space-time attention) | " |

`common.py` handles data, features, the extra training windows and metrics. `deep.py` is the torch loop and `gbm.py` the LightGBM search. Key shared input: a per-cell **seasonal harmonic fill rate** (sine/cosine fit of each cell's training-year fill rate against day of year, pulled toward the harbour average; training rows leave their own year out).

Deep setup: 1,835 training maps (the 367 original windows plus extra train-only windows with 1/2/3-survey input spacing and 7–9-survey horizons), depth fed **relative to the 15×15 neighbourhood**, EMA of the weights, best checkpoint by val every 100 steps (1,500 steps). The models predict the residual over climatology.

## Results (after optimization)

| model | val MAE | test MAE | test hot | test RMSE | test AUROC | test AP | vs persistence |
|---|---|---|---|---|---|---|---|
| **m02 LightGBM** | 0.548 | **0.447** | **0.462** | **0.668** | 0.894 | 0.722 | **36%** |
| m03 hurdle LightGBM | **0.546** | 0.451 | 0.474 | 0.671 | **0.894** | **0.730** | 35% |
| m10 TimeSformer | 0.589 | 0.491 | 0.562 | 0.731 | 0.875 | 0.651 | 30% |
| m04 damped Holt | 0.601 | 0.497 | 0.543 | 0.746 | 0.873 | 0.633 | 29% |
| m08 SimVP | 0.581 | 0.499 | 0.535 | 0.736 | 0.873 | 0.649 | 28% |
| m07 ConvGRU | 0.582 | 0.502 | 0.545 | 0.741 | 0.866 | 0.652 | 28% |
| m06 U-Net | 0.590 | 0.506 | 0.593 | 0.743 | 0.873 | 0.651 | 27% |
| m01 spline GAM | 0.591 | 0.510 | 0.640 | 0.738 | 0.872 | 0.609 | 27% |
| m05 EOF reduced-rank | 0.580 | 0.522 | 0.603 | 0.772 | 0.858 | 0.637 | 25% |
| m09 FNO | 0.579 | 0.531 | 0.658 | 0.776 | 0.862 | 0.606 | 24% |
| baseline: harmonic rate | 0.606 | 0.521 | 0.586 | 0.764 | 0.858 | 0.596 | 25% |
| baseline: persistence | 0.773 | 0.698 | 1.106 | 1.058 | 0.500 | 0.201 | 0% |

## Before → after

| model | val MAE | test MAE | main change |
|---|---|---|---|
| m01 | 0.593 → 0.591 | 0.530 → **0.510** | splines + climatology×driver interactions, residual target |
| m02 | 0.560 → **0.548** | 0.448 → 0.447 | small trees (31 leaves), Huber loss, residual target, top-3 ensemble |
| m03 | 0.557 → **0.546** | 0.443 → 0.451 | tuned params, τ = 1 ft, expected-value combine |
| m04 | 0.606 → 0.601 | 0.521 → **0.497** | 12-survey history instead of 4 |
| m05 | 0.574 → 0.580 | 0.626 → **0.522** | local relative depth, train-CV selection, extra windows |
| m06 | 0.591 → 0.590 | 0.525 → **0.506** | extra windows, local depth, EMA |
| m07 | 0.600 → **0.582** | 0.520 → **0.502** | " |
| m08 | 0.604 → **0.581** | 0.521 → **0.499** | " |
| m09 | 0.604 → **0.579** | 0.521 → 0.531 | " (overfits fastest) |
| m10 | 0.600 → **0.589** | 0.521 → **0.491** | " |

## Takeaways

- **The harbour got deeper.** Compared with the 2014–22 average, the channel is ~2 ft deeper in 2023–24 and ~3.8 ft deeper in 2025–26. Models given absolute depth extrapolate badly: the old EOF had a −0.27 ft bias and 0.626 test MAE. Feeding depth relative to the neighbourhood fixed EOF and was the change that let the deep models learn (with absolute depth, the U-Net's val error rose as it trained; with local depth it kept falling). Trees can't extrapolate, and dropping absolute depth hurt LightGBM on val, so the tabular models keep it.
- **More training windows help the deep models, not the trees.** They took the deep models from "no better than climatology" to 3–6% better than it on test. LightGBM's val got slightly worse with them, so the tree models use only the original windows.
- **Gradient boosting is still clearly best**, and val-only tuning improved val for both boosting models. Test barely moved (m02) or slipped (m03 0.443 → 0.451): the 2023–24 and 2025–26 years behave differently, so gains on val don't fully carry over. Averaging the top 3 configs made m02 the steadiest.
- **TimeSformer is now the best deep model** on test, and every deep model except FNO now beats climatology. FNO overfits within ~500 steps (train loss falls to 0.28).
- **A longer history makes Holt useful.** Smoothing over 12 surveys denoises the current depth, and the trend now gets weight 0.3. Over 4 surveys the trend was pure noise.
- All results use one seed. Treat differences under ~0.01 ft as noise.

## On the dredging map

`publish_frontend.py` adds the LightGBM ensemble (m02) and TimeSformer (m10) to the platform's dredging page as two extra models in `backend/static/data/anchorage/manifest.json`. It reuses `tools/anchorage`'s footprint, GeoJSON writer, comparison export and verifier, and leaves the XGBoost and physics files untouched. Run `m02_lightgbm.py` and `m10_timesformer.py` first; they save their models to the git-ignored `artifacts/`.

```
pip install pyproj xgboost threadpoolctl
python m02_lightgbm.py && python m10_timesformer.py && python publish_frontend.py
```

The page shows the platform's metric: per-survey test MAE averaged over the 71 test windows, on the platform core (cells observed in ≥80% of training maps, dredged cells included). Persistence reproduces the platform's 0.706.

| model | platform test MAE (ft) |
|---|---|
| XGBoost + neighborhoods (Alex) | 0.422 |
| LightGBM ensemble | 0.459 |
| TimeSformer | 0.504 |
| Physics-informed (Alex) | 0.518 |
| Persistence | 0.706 |
