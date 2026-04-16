# Bucket Optimization + GBR Inference

## Overview

Two objectives:
1. Find the optimal number of bleaching severity classes by recovering continuous `Percent_Bleaching` from the GCBD CSV and testing finer bucket schemes
2. Run real-time inference on 133 GBR reef cluster centroids from April 2026

All code lives in `xgboost-model/experiments/buckets/`. Nothing in `src/` or `outputs/` was modified.

---

## Step 1: Recovering Continuous Bleaching Values

The NPZ file (`sequences_reduced_16.npz`) only stores integer class labels (0, 1, 2), not the original continuous `Percent_Bleaching` values. To test finer buckets, we needed to recover those values.

**Method:** Match each NPZ sample to the GCBD CSV (`global_bleaching_environmental.csv`) using the `(lat, lon, year, month)` key stored in `meta`. The `extract_sequence.py` pipeline stores the original GCBD survey coordinates (not CRW grid coordinates) in meta, so matching at 3-decimal precision works.

**Result:** 26,445 / 28,539 samples recovered (92.7%). Binning agreement with original labels: 99.96% (11 mismatches out of 26,445).

### Continuous Distribution

| Range | Count |
|-------|-------|
| 0% | 12,658 |
| 0–1% | 2,673 |
| 1–5% | 3,531 |
| 5–25% | 4,112 |
| 25–50% | 1,601 |
| 50–75% | 1,386 |
| 75–100% | 484 |

---

## Step 2: Bucket Scheme Comparison

Tested 5 schemes with XGBoost (raw-weekly flattening, 68 features, tuned hyperparameters, balanced class weights, 70/15/15 stratified split).

| Scheme | Classes | Macro F1 | Weighted F1 | Min Class % |
|--------|---------|----------|-------------|-------------|
| 3_original (0/1-50/51-100) | 3 | 0.756 | 0.834 | 7.1% |
| 4_split_mod (0/1-25/26-50/51-100) | 4 | 0.672 | 0.801 | 6.1% |
| 4_split_sev (0/1-50/51-75/76-100) | 4 | 0.676 | 0.826 | 1.8% |
| 5_granular (0/1-25/26-50/51-75/76-100) | 5 | 0.606 | 0.789 | 1.8% |
| **4_ecological (0-5/6-30/31-100)** | **3** | **0.776** | **0.862** | **12.0%** |

### Key Findings

1. **Ecological binning wins.** The `[-1, 5, 30, 100]` scheme (Healthy / Stressed / Severe) achieves the best macro F1 (0.776 vs 0.756 for the original 3-class scheme) — a **+0.020 improvement**.

2. **More classes hurt.** Every 4-class and 5-class scheme dropped macro F1 by 0.08–0.15. The moderate and severe sub-splits create classes too small and too similar for XGBoost to separate cleanly.

3. **The ecological boundaries are better calibrated.** The original bins (0% / 1–50% / >50%) are arbitrary. The ecological bins (0–5% / 6–30% / >30%) align with marine biology thresholds:
   - **0–5%**: Effectively no bleaching — background noise
   - **6–30%**: Stress signal — coral is responding to thermal pressure
   - **>30%**: Mass bleaching — reef-wide mortality risk
   
   This puts the boundary between "healthy" and "stressed" at 5% (not 1%), which avoids treating trace bleaching as a positive signal. And it lowers the "severe" threshold to 30% (not 50%), catching events that are already ecologically devastating.

4. **Class balance improves.** The ecological scheme has 71.3% / 16.7% / 12.0% distribution vs the original 58.0% / 35.0% / 7.1%. The severe class nearly doubles in size (from 7.1% to 12.0%), giving the model more positive examples to learn from.

### Why Ecological Beats Original

The original 3-class scheme treats 1% and 49% bleaching as the same class ("Moderate"). That's a huge range — a reef with 1% bleaching is fine, a reef with 49% bleaching is in crisis. By splitting at 5% and 30% instead:
- The model no longer has to reconcile "barely bleached" and "almost dead" in the same class
- The severe class grows from 1,870 to 3,177 samples (70% larger), reducing the class imbalance problem
- The healthy class absorbs trace bleaching (1–5%), which is noise rather than signal

---

## Step 3: Final Model (Ecological Binning)

| Metric | Healthy (0-5%) | Stressed (6-30%) | Severe (>30%) |
|--------|---------------|-----------------|---------------|
| Precision | 0.951 | 0.618 | 0.712 |
| Recall | 0.911 | 0.678 | 0.796 |
| F1 | 0.931 | 0.646 | 0.752 |
| Support | 2,830 | 661 | 476 |

**Overall: Macro F1 = 0.776, Accuracy = 85.8%**

The model is strongest at identifying healthy reefs (93.1% F1) and detects severe bleaching well (75.2% F1, 79.6% recall). The stressed class is harder (64.6% F1) — it's the transitional zone where thermal stress is present but not yet devastating.

---

## Step 4: GBR Inference (April 2026)

Ran the ecological-binned model on 133 GBR reef cluster centroids from CRW satellite data (week ending 2026-04-12).

### Results

**All 133 clusters predicted as Healthy (0-5%).** No change between year=2026 and year=2020 (clamped).

This is expected:
- **April is early autumn in the Southern Hemisphere** — bleaching season peaks in Feb/Mar when SSTs are highest
- **DHW values are low** — max across all clusters is 5.6°C-weeks (mass bleaching typically requires DHW > 8)
- **Severe probabilities are negligible** — highest P(severe) is 2.4% (Cluster 37 near Cairns)

### Highest Risk Clusters (Still Low Risk)

| Cluster | Location | DHW Max | P(Severe) |
|---------|----------|---------|-----------|
| 37 | -16.94, 145.99 (Cairns) | 4.9 | 2.4% |
| 70 | -16.38, 145.56 (N. Qld) | 5.1 | 2.3% |
| 44 | -14.45, 144.91 (Cooktown) | 5.5 | 2.3% |
| 46 | -14.13, 144.50 (Cape Tribulation) | 5.6 | 2.1% |

All in the northern GBR between Cooktown and Cairns — the warmest section of the reef, consistent with known bleaching hotspots.

### Year Extrapolation

Zero clusters changed class when year was clamped from 2026 to 2020. The satellite features (low DHW, moderate SST) dominate the prediction — the year feature doesn't matter when thermal stress is clearly below bleaching thresholds.

---

## Files

| File | Description |
|------|-------------|
| `analyze_buckets.py` | Step 1: recover continuous values, test 5 bucket schemes |
| `train_best.py` | Step 2: train final model with winning scheme |
| `inference.py` | Step 3: run GBR inference on 133 centroids |
| `results/bucket_comparison.csv` | Macro/weighted F1 for all 5 schemes |
| `results/winner_config.json` | Winning scheme config |
| `results/best_model.json` | Trained XGBoost model (ecological binning) |
| `results/model_config.json` | Feature columns and class labels for inference |
| `results/confusion_matrix.png` | Confusion matrix (counts + row-normalized %) |
| `results/feature_importance.png` | Top 20 features by gain |
| `results/gbr_predictions.csv` | Full predictions for 133 clusters |
| `results/gbr_risk_map_data.csv` | Simplified output for mapping |
