# Anchorage model overlays

The `alex-wang` branch provides separate training scripts and a shared map output for two Anchorage models.

## Models and measured results

The target is the change from the last observed depth map to a map 28 days later. Depth is positive downward below MLLW. Models calculate in metres and export feet.

| Model | Inputs | Test MAE, ft |
| --- | --- | ---: |
| XGBoost + neighborhoods | 60 features | 0.4328564 |
| Physics-informed source + diffusion | 84 features | 0.5416623 |
| Persistence: retain the last measured depth | Last map | 0.7062322 |

MAE is averaged over cells within each forecast window, then over windows, so large surveys do not dominate. XGBoost reduces error by about 39% relative to persistence; the physics model reduces it by about 23%. XGBoost is the deployment default: it had the lowest measured test MAE among the benchmark candidates and a small artifact. Its result is close to CatBoost (0.434970 ft), so the small difference is not evidence of a definitive statistical advantage. See `benchmark_results.csv` and `release_comparison.json`.

- `train_xgboost.py` trains our own XGBoost adapter. It uses four earlier surveys, cell location, depth differences and trend, survey gaps, season, past dredging flags, a training-derived baseline, and neighborhood summaries at 30, 90 and 210 m. Its 29 raw predictors become 58 value/missingness columns, plus curvature and horizon. It does not use weather or river drivers.
- `train_physics.py` independently trains our physics adapter. Its 42 raw predictors comprise 14 survey/history predictors plus 7- and 28-day summaries and coverage for tides, three rivers, rain, snowfall and temperature. Each has a missingness flag, giving 84 neural-network inputs. Drivers stop before the origin date; no future observed weather, flow or dredging flags enter either model.

## Data handling

The reproducible training source is the pinned `timeseries_v2/timeseries.npz` at commit `c9e6f432b581aea2fd435618f72522ca7f784167`. `--dataset` accepts a local copy and checks its exact Git blob; omitting it downloads that copy. The supplied split is chronological: 2014–2022 training, 2023–2024 validation, and 2025–2026 test. Validation selects boosting checkpoints or neural checkpoints; test data do not fit preprocessing or weights. The driver archive includes retrospective weather reanalysis and date-only alignment, so these are hindcast results rather than validation of a live operational feed.

The spatial core requires observations in at least 80% of training surveys. Core selection, historical-rate baseline, median imputers and scalers use training data only. Training samples up to 400 measured cells per window. Missing origin or target depths exclude a training/scoring pair; predictions require a measured origin. **Neither target maps nor origin maps are gap-filled.** Older predictors and driver gaps use training medians with explicit missingness flags. Driver summaries need at least 80% daily coverage; coverage is also an input. This preserves missing observations while allowing incomplete histories, but cannot establish accuracy in unsurveyed cells or untested winter conditions.

The attachment contributes only its 5,500-cell geometry footprint, never prediction values or labels. Every published frame uses the intersection of that footprint and both models' valid predictions. Cells may therefore be omitted; both models show exactly the same cells for a given date pair. The latest frame exports 5,483 cells: 17 reference cells fail the training-only coverage criterion. The September 14 origin exports 5,432 because another 51 origin depths are missing.

## What the physics constraint means

The two-seed ensemble uses a reduced bed-evolution update:

```text
Δd = (T / 28) × (baseline28 + σ fθ(X)) + κ T Ld
predicted_depth = observed_origin_depth + Δd
```

Here `d` is positive-down depth in metres, `T` is days, `baseline28` is the training-derived 28-day change, `σ` is the training residual scale, and `fθ` is a small `84 → 32 → 32 → 1` tanh network. `Ld` sums four-neighbor depth differences divided by `(10 m)²`. Diffusion smooths local bed differences; `κ ≥ 0` is bounded for a stable transport-only update. The fitted members have `κ = 0.159298` and `0.160192 m²/day`.

Neighbor exchange is equal and opposite. Missing/non-core edges have zero flux, so **only the diffusion term is conservative on the observed core**. This is a computational boundary assumption, not a measured harbor boundary condition. The learned source remains unconstrained and can absorb sediment deposition, erosion or dredging. This is feasible as a physics-informed surrogate; a calibrated sediment-transport PINN is not identifiable from these inputs alone. That would require sediment concentrations/properties, currents and boundary fluxes, plus independently validated dredging/placement terms.

## Rebuild and publish

Run from the repository root with Python 3.10 or 3.11 (verified here on 3.10.12). PyTorch is needed only for physics training.

```bash
python3.11 -m venv tools/anchorage/.cache/venv
source tools/anchorage/.cache/venv/bin/activate
python -m pip install -r tools/anchorage/requirements.txt
python -m pip install -r tools/anchorage/requirements-physics.txt
python tools/anchorage/train_xgboost.py
python tools/anchorage/train_physics.py
python tools/anchorage/publish_overlays.py --frames 6
python tools/anchorage/verify_overlays.py
```

Both trainers accept `--dataset /path/to/pinned-timeseries.npz` and `--output /path/to/artifact.joblib`. Defaults save to the Git-ignored `.cache/` directory. Publishing accepts `--artifacts /path/to/artifact-directory`. Pass `--reference /path/to/anchorage_prediction.geojson` to the verifier to check containment within the supplied geometry.

Publishing `--dataset` is the common pinned input. `--xgboost-input` and `--physics-input` override each adapter's inference archive independently, so teammates need not use identical inputs internally. These archives must satisfy the current adapter's v2 schema, trained grid and required historical dates; the physics archive also needs its daily-driver arrays. Different schemas need their own adapter, not forced conversion into the XGBoost feature set.

The default release contains six historical **28-day hindcasts**, each made from its own earlier origin. The slider moves among those date pairs; it does not imply evaluated forecasts at arbitrary horizons. `--origin YYYY-MM-DD` can export one 28-day projection from a compatible archive, subject to available history and evaluated seasonal coverage.

## Shared output and deployment

`publish_overlays.py` registers adapters in `ADAPTERS` and writes `backend/static/data/anchorage/manifest.json`. The manifest lists models as `{id, label, mae_ft}` and frames as `{id, from_survey, predicted_for, outputs}`. `outputs` maps model IDs to local GeoJSON URLs. The adjacent `evaluation.json` records training scores, feature names, XGBoost split gain and physics coefficients and checkpoints. To add a model, supply its trainer, input loader and prediction function, then register its artifact and model ID. The map reads the model list from the manifest.

Every GeoJSON follows the attachment's exact property keys:

```json
{
  "type": "FeatureCollection",
  "properties": {"from_survey": "2026-09-17", "predicted_for": "2026-10-15"},
  "features": [{
    "type": "Feature",
    "geometry": {"type": "Polygon", "coordinates": [[[0,0],[1,0],[1,1],[0,1],[0,0]]]},
    "properties": {"depth_ft": 40.0, "change_ft": -0.5}
  }]
}
```

Coordinates above illustrate structure only; exports contain WGS84 longitude/latitude corners of actual 10 m cells. Each polygon has four distinct corners and a repeated first corner. `depth_ft` is origin depth plus predicted change; negative `change_ft` means shallower. No model-specific fields are added to these properties. The verifier checks dates, finite numbers, closed rings, duplicates, common model footprints, local asset paths and optional containment within the attachment.

Publication writes all GeoJSON assets before switching the manifest. The existing Flask website serves those static assets on `/dredging`; serving the page requires neither PyTorch nor XGBoost and performs no online training. Generated static files are the deployable result; model artifacts and downloaded training data remain in the ignored cache.

For a local preview, install `backend/requirements.txt` in the website environment and run the existing Flask app.

```bash
cd backend
SECRET_KEY=local-preview-only FAST_START_FLOOD_DATA=1 python -m flask --app run:app run --port 5076
```

Open `http://127.0.0.1:5076/dredging` and use **Skip Login** if prompted.
