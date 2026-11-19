# Flood label audit

## New training pipeline

`train.py` is a new research trainer addressing the old V2 preprocessing and
evaluation problems. It does not overwrite or change the website's V3/V9 models.
No real-data training or accuracy claim has been made. The original labels and
prepared input files still need to be recovered or rebuilt.

Implemented changes:

- Chronological partitions with entire input-to-target windows purged at boundaries.
- Explicit event groups: a storm and its linked controls cannot cross partitions.
- Optional completely withheld test regions, in addition to the time holdout.
- Separate precipitation/soil scalers fitted only to training samples.
- No augmentation for validation or testing; optional noise only on training inputs.
- Independent CNN/LSTM temporal encoders for 32 three-hour precipitation frames
  and 14 daily soil frames, with actual frame age in each sequence. All frames
  are used; the old `min(T_precip, T_soil)` truncation is not reproduced.
- Validation loss selects the checkpoint. Test evaluation is a separate command.
- Input timestamps and publication times must precede forecast issuance.
- Frozen sample IDs, event groups, configuration, and file hashes are recorded.
- Evaluation reports precision, recall, PR curve, average precision, Brier score,
  confusion matrix, per-sample predictions, and a training-prior baseline.

### Prepared input contract

The audit CSV above is **not** directly usable as a training manifest. Create a
manifest using `manifest.example.csv` only after labels and satellite windows
have been reviewed. Each row requires:

| Column | Meaning |
|---|---|
| `sample_id` | Unique example identifier |
| `event_group` | Shared storm/episode identifier across countries and paired controls |
| `region` | Region used for geographic holdouts |
| `issued_at` | UTC forecast issuance timestamp |
| `history_start` | Start of the complete input interval |
| `target_start`, `target_end` | Explicit future label interval |
| `label` | 1 for flood onset in that interval, 0 for a reviewed control |
| `npz_path` | Prepared NPZ path relative to the manifest |

Use timezone-aware ISO timestamps, such as `2020-06-01T00:00:00Z`.
`target_start - issued_at` must match `--lead-days` exactly. Do not substitute
an event date for issuance. Label 0 must not mean merely absent from an archive.
The grouping is supplied by the data curator: code cannot infer that two distant
records are one storm, or validate an unverified negative.

Each NPZ must contain these arrays (no object arrays/pickles):

- `precip`: `(32, H, W)` float32, finite nonnegative **3-hour accumulation in mm**.
- `soil`: `(14, H, W)` float32, **daily mean volumetric fraction**; NaN is allowed
  in individual soil cells, with a missingness mask added by the trainer.
- `precip_times`, `soil_times`: ISO string vectors of **interval-end** timestamps,
  in strictly regular 3-hour / 24-hour order respectively.
- `precip_available_at`, `soil_available_at`: matching ISO string vectors recording
  when those inputs actually became available. Do not invent these timestamps to
  make a retrospective product pass a live-forecast check.

No missing days are silently skipped. Entirely missing frames fail validation.
All patches must have consistent dimensions per modality, at least 8x8. The
physical CRS, grid spacing, orientation, and centering must also be standardized
by feature preparation; shape checks do not establish geographic alignment.
Changing the source product or forecast lead requires a newly evaluated recipe.
This pipeline does not download or aggregate raw Earthdata observations.

### Commands

Install GPU-enabled PyTorch separately if CUDA is wanted; `requirements.txt`
does not guarantee a CUDA build on Windows. Date cutoffs below are examples:

```powershell
python -m pip install -r tools/flood/requirements.txt
python tools/flood/train.py train --manifest tools/flood/.local/manifest.csv --output tools/flood/.local/run-42 --validation-start 2019-01-01T00:00:00Z --test-start 2021-01-01T00:00:00Z --lead-days 5 --seed 42
# Only after freezing the recipe, evaluate its held-out partition:
python tools/flood/train.py evaluate --manifest tools/flood/.local/manifest.csv --checkpoint tools/flood/.local/run-42
python -m unittest discover -s tools/flood -p "test_*.py"
```

For a regional holdout, add `--test-regions region_name,another_region` to training.
Those regions are excluded from training/validation, and only those regions are
included in the later test period. Every partition must retain both classes.
Use different output directories for different seeds. Compare seeds/configurations
on validation before running test evaluation. A previously inspected test period
does not become independent again merely by using this trainer.

Training saves `model.pt`, `scalers.joblib`, `splits.json`, and `training.json`.
Test evaluation writes `test_metrics.json`, without changing the checkpoint.
Threshold is fixed at 0.5; scores are not calibrated probabilities of real-world
flood prevalence merely because the training classes are balanced. Bootstrap
uncertainty should use event groups, not individual overlapping patches.

This architecture is intentionally incompatible with old inference weights.
Deployment needs a matching input adapter and independent validation; do not
rename its checkpoint to V3/V9 or change the serving import to load it.

## Label preparation

The original label CSV is not in the checkout. This tool prepares a reviewable
candidate dataset once it is recovered; it does not claim to repair unseen labels.
No downloads, neural training, or changes to the deployed model are performed.

Run from the repository root (dates below are examples, not selected splits):

```powershell
python tools/flood/prepare_labels.py path/to/CompleteFloodDataset.csv --validation-start 2019-01-01 --test-start 2021-01-01 --require-smap
python -m unittest discover -s tools/flood -p test_prepare_labels.py
```

Required columns match Dean's V2 trainer: `ID`, `Country`, `lat`, `long`,
`Began` (ISO date), and `isFlood` (0 or 1). Optional `Ended` defines event duration.
Preserve original source identifiers and add `event_group` to link records from
one storm across countries and paired controls. Without that field the tool uses
`ID`, which cannot establish that distinct records are independent.

Outputs default to ignored `.local/accepted.csv`, `quarantine.csv`, and
`report.json`. The report records input hash, split counts, screening parameters,
and remaining limitations. Source data is not overwritten.

The tool quarantines invalid rows, duplicate location/date/label observations,
events spanning partition boundaries, and negatives near recorded floods.
The default 100 km / 14-day buffers are screening heuristics, not validated flood
extent or negative-sampling rules. Review them for the catchment and flood type.

Unverified negatives are quarantined. Set `negative_verified=yes` only after
documented independent review; do not bulk-mark rows simply to make training run.
Absence from a major-disaster archive is not evidence of no flooding. This tool
does not independently check external archives or certify accepted labels.

With `--require-smap`, the audit requires 22 days of history after the March 31,
2015 product start (matching the old trainer's lookback). Feature assembly must
also check actual file coverage, issue-time availability, spatial alignment,
forecast lead, and full input-to-target boundaries. A date cutoff alone does not
guarantee the data exists or would have been available to a forecaster.

Before training, review nearby cross-partition event warnings, ensure both labels
exist in every partition, and freeze the grouping/split policy. Fit imputers and
scalers on training rows only; use augmentation only during training. The old V2
training script has not been rewritten or retrained by this label audit.
