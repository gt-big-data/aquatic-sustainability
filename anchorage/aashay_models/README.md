# Anchorage Harbor: 10 depth-change models

Same task, data and splits as `anchorage_convlstm_drivers.ipynb`: four depth maps in, change in each 10 m cell ~4 weeks after the last map out (`timeseries_v2`, train 2014–22, val 2023–24, test 2025–26). Scored with masked MAE in ft. Hot-spot MAE covers cells that filled more than 0.75 ft per 4 weeks on average in the training years. AUROC/AP rank cells that shoal by more than 1 ft.

```
pip install numpy scipy scikit-learn lightgbm torch pandas tabulate
cd anchorage/aashay_models
python m00_baselines.py && python m01_linear.py ... && python summarize.py
```

| # | file | model | type |
|---|------|-------|------|
| 1 | `m01_linear.py` | Ridge / Huber regression (picked on val) | common |
| 2 | `m02_lightgbm.py` | LightGBM, L1 loss | common |
| 3 | `m03_hurdle.py` | Hurdle: LightGBM shoal/stable/deepen classifier × per-class L1 regressors | niche |
| 4 | `m04_holt.py` | Damped-trend Holt exponential smoothing per cell + seasonal climatology | common TS |
| 5 | `m05_eof_rrr.py` | EOF (PCA) reduced-rank regression on whole maps | niche (geoscience) |
| 6 | `m06_unet.py` | U-Net | common |
| 7 | `m07_convgru.py` | 2-layer ConvGRU | common |
| 8 | `m08_simvp.py` | SimVP (CNN encoder, inception translator, decoder) | niche |
| 9 | `m09_fno.py` | Fourier Neural Operator | niche |
| 10 | `m10_timesformer.py` | TimeSformer-lite (divided space-time attention on 8×8 patches) | niche |

`common.py` loads the data and builds the per-cell features and metrics. `deep.py` holds the shared torch training loop. Inputs: the 4 maps, observed/dredged flags, survey gaps, neighbour stats, the tide/river/rain/temperature drivers, and a per-cell **seasonal harmonic fill rate** (sine/cosine fit of each cell's training-year fill rate against day of year, pulled toward the harbour average, with the sample's own year left out for training rows). The deep models predict the residual on top of that climatology and start from it (zero-initialised head).

## Results

| model | val MAE | val hot | test MAE | test hot | test RMSE | test AUROC | test AP | vs persistence |
|---|---|---|---|---|---|---|---|---|
| **m03 hurdle LightGBM** | **0.557** | 0.658 | **0.443** | **0.451** | **0.669** | **0.891** | **0.735** | **37%** |
| m02 LightGBM | 0.560 | 0.660 | 0.448 | 0.461 | 0.673 | 0.885 | 0.721 | 36% |
| m07 ConvGRU | 0.600 | 0.728 | 0.520 | 0.583 | 0.764 | 0.857 | 0.595 | 26% |
| m04 damped Holt | 0.606 | 0.737 | 0.521 | 0.586 | 0.764 | 0.858 | 0.596 | 25% |
| m08 SimVP | 0.604 | 0.735 | 0.521 | 0.585 | 0.764 | 0.858 | 0.596 | 25% |
| m09 FNO | 0.604 | 0.733 | 0.521 | 0.586 | 0.764 | 0.858 | 0.596 | 25% |
| m10 TimeSformer | 0.600 | 0.730 | 0.521 | 0.585 | 0.765 | 0.857 | 0.595 | 25% |
| m06 U-Net | 0.591 | 0.714 | 0.525 | 0.585 | 0.767 | 0.857 | 0.594 | 25% |
| m01 linear (Huber) | 0.593 | 0.702 | 0.530 | 0.633 | 0.769 | 0.854 | 0.590 | 24% |
| m05 EOF reduced-rank | 0.574 | 0.601 | 0.626 | 0.671 | 0.924 | 0.832 | 0.578 | 10% |
| baseline: harmonic rate | 0.606 | 0.737 | 0.521 | 0.586 | 0.764 | 0.858 | 0.596 | 25% |
| baseline: 3-season rate | 0.650 | 0.752 | 0.634 | 0.792 | 0.893 | 0.758 | 0.495 | 9% |
| baseline: persistence | 0.773 | 1.252 | 0.698 | 1.106 | 1.058 | 0.500 | 0.201 | 0% |

Full table in `results/summary.csv`, per-model settings in `results/*.json`. For reference: refill model 0.591 / 0.478, GNN 0.665 / 0.584, ConvLSTM + drivers 0.689 / 0.595 (val / test).

## Takeaways

- **Gradient boosting wins clearly.** Hurdle LightGBM is the best on every test metric, including hot spots (0.451 vs 0.509 for the refill model). Top LightGBM features: harmonic climatology (cell and 5×5 neighbourhood), current depth, position, Ship Creek flow, bed gradient.
- **The seasonal harmonic climatology is most of the signal.** Alone it reaches 0.521 test MAE, already ahead of the ConvLSTM.
- **Past trend adds nothing.** Holt's best fit on train was trend weight 0 and alpha 1, so it collapses to the climatology baseline: the change over the last ~3 weeks doesn't predict the next 4 weeks.
- **The deep models don't beat climatology.** All five reach their best val score after epoch 1 and then overfit (FNO train loss 0.53 → 0.37 while val went 0.60 → 0.70). 367 training maps is too few for them. I also tried random crops, removing the position channels, lower learning rates, and leaving GroupNorm out (that last one is kept); none helped much. Since they start from the climatology, their numbers stay close to it.
- **EOF reduced-rank regression overfits to the val years** (best val, worst test among the learned models).
- All results use one seed. Treat differences under ~0.01 ft as noise.
