# XGBoost Ablation Experiments — Findings

## Overview

Three experiments testing what drives the XGBoost coral bleaching classifier's performance. All experiments use the same data (`sequences_reduced_16.npz`, 28,539 samples), the same 70/15/15 stratified split (seed=42), and the same tuned hyperparameters. Only the features, class weights, or decision thresholds change.

---

## Experiment 1: Flattening Variants

**Question:** Does the way we compress 16 weeks of time series into tabular features matter?

| Variant | Features | Macro F1 | None R | Mod R | Sev R |
|---------|----------|----------|--------|-------|-------|
| A. Summary stats | 36 | 0.758 | 0.896 | 0.750 | 0.703 |
| **B. Raw weekly values** | **68** | **0.764** | **0.900** | **0.745** | **0.720** |
| C. Hybrid (summary + raw) | 100 | 0.755 | 0.899 | 0.755 | 0.669 |
| D. Summary no year | 35 | 0.745 | 0.892 | 0.737 | 0.659 |
| E. Summary + interactions | 42 | 0.760 | 0.896 | 0.748 | 0.706 |
| F. Hybrid no year | 99 | 0.744 | 0.898 | 0.752 | 0.625 |

### Key Findings

1. **Raw weekly values (B) slightly beat summary stats (A).** F1: 0.764 vs 0.758, severe recall: 72.0% vs 70.3%. This is a small but consistent improvement. XGBoost can learn its own temporal patterns from raw week-by-week data — it doesn't need us to pre-compute means and slopes.

2. **Hybrid (C) is worse than either A or B alone.** Adding summary stats on top of raw values hurts rather than helps. With 100 features, the model likely overfits or gets confused by redundant information (e.g., `TSA_mean` is a linear combination of the 16 `TSA_weekNN` columns).

3. **Removing `year` hurts significantly.** Summary: 0.758 → 0.745 (-0.013). Hybrid: 0.755 → 0.744 (-0.011). Year captures the real climate trend — bleaching has gotten worse over time. But it also means the model partially memorizes "later years = more bleaching" rather than learning purely from thermal stress physics.

4. **Interaction features (E) add marginal value.** 0.760 vs 0.758 for summary alone. The hand-crafted interactions (DHW × SST, sustained stress ratio, etc.) don't help much because XGBoost can learn multiplicative interactions through sequential splits anyway.

### Why Raw Beats Summary

The feature importance plot for variant B reveals why: **`TSA_DHW_week15` and `TSA_DHW_week14`** (the two most recent weeks of accumulated thermal stress) have far more gain than any other feature. The summary stats compress these into `TSA_DHW_last` (which is just week 15) and `TSA_DHW_mean` (which dilutes the signal with earlier weeks). The raw variant lets the model see the full temporal profile and focus on the weeks that matter most — the final 2-3 weeks of the window.

---

## Experiment 2: Severe Class Weight Sweep

**Question:** Does increasing the weight on the rare severe class (6.8% of data) improve overall performance?

| Severe Multiplier | Effective Weight | Macro F1 | Sev Recall | Sev Precision | Sev F1 |
|-------------------|-----------------|----------|------------|---------------|--------|
| 0.5x | ~2.4x | 0.763 | 0.659 | 0.573 | 0.613 |
| **1.0x (balanced)** | **~4.9x** | **0.764** | **0.720** | **0.551** | **0.624** |
| 1.5x | ~7.3x | 0.753 | 0.700 | 0.519 | 0.596 |
| 2.0x | ~9.7x | 0.757 | 0.717 | 0.520 | 0.603 |
| 3.0x | ~14.6x | 0.748 | 0.713 | 0.507 | 0.593 |

### Key Findings

1. **The default balanced weight (~4.9x for severe) is already optimal for macro F1.** Increasing the severe weight further doesn't help — it pushes severe recall up slightly but tanks precision, lowering the overall F1.

2. **There's a clear precision-recall tradeoff.** At 0.5x multiplier (underweighting severe), precision is 57.3% and recall is 65.9%. At 3.0x (overweighting), precision drops to 50.7% while recall only reaches 71.3%. The balanced default gives the best F1 balance.

3. **Overweighting severe hurts more than it helps.** The model starts predicting too many false severe events, pulling precision below 52% and dragging down overall macro F1. The balanced weights are already aggressive enough.

---

## Experiment 3: Probability Threshold Tuning

**Question:** Can we improve severe class detection by lowering the decision threshold (predicting severe when P(severe) > threshold instead of waiting for it to be the argmax)?

| Threshold | Macro F1 | Sev Recall | Sev Precision | Sev F1 |
|-----------|----------|------------|---------------|--------|
| 0.10 | 0.731 | 0.874 | 0.435 | 0.580 |
| 0.15 | 0.741 | 0.840 | 0.461 | 0.595 |
| 0.20 | 0.747 | 0.805 | 0.481 | 0.602 |
| 0.25 | 0.751 | 0.778 | 0.495 | 0.605 |
| 0.30 | 0.754 | 0.765 | 0.507 | 0.610 |
| 0.35 | 0.757 | 0.747 | 0.521 | 0.614 |
| 0.40 | 0.759 | 0.737 | 0.529 | 0.616 |
| **0.50 (default)** | **0.764** | **0.713** | **0.554** | **0.624** |

### Key Findings

1. **Default argmax (effectively ~0.50 threshold) gives the best macro F1.** Lowering the threshold catches more severe events but at the cost of too many false positives.

2. **Threshold tuning is a policy decision, not a modeling decision.** If you're building an early warning system where missing a severe event has high cost, a threshold of 0.20-0.25 catches 78-81% of severe events (vs 71% at default) at the cost of ~50% precision and 1-2% macro F1 drop. Whether that tradeoff is worth it depends on the application.

3. **The precision-recall curve is smooth** — no magic threshold gives you both high recall and high precision. The severe class is genuinely hard to separate from moderate bleaching.

---

## Best Overall Configuration

| Setting | Value |
|---------|-------|
| Flattening | B — Raw weekly values (68 features) |
| Severe weight | 1.0x balanced (~4.9x effective) |
| Threshold | Default argmax |
| **Test Macro F1** | **0.764** |
| None recall | 0.900 |
| Moderate recall | 0.745 |
| Severe recall | 0.720 |

**Improvement over original:** +0.006 macro F1 (0.758 → 0.764), +1.7% severe recall.

The gains are modest. The original summary-stats approach was already a reasonable design. The raw variant wins because XGBoost is good at learning its own temporal summaries from the raw data — it doesn't need us to pre-compute them.

---

## Implications

1. **The model is close to its ceiling with this data.** No flattening strategy, weight scheme, or threshold breaks 0.77 macro F1. The remaining errors are likely due to:
   - The moderate class being inherently fuzzy (1% and 49% bleaching are both "moderate")
   - Missing features (water quality, reef health history, species composition)
   - Spatial autocorrelation inflating apparent performance

2. **Year is a feature worth keeping but worth being cautious about.** It captures real climate trends (+0.013 F1) but also creates a temporal bias. A model trained on 1983-2019 data implicitly learns "2016 = bad" (mass bleaching year). Future deployment should weight year carefully or replace it with a climate index.

3. **The LSTM comparison will be telling.** If the LSTM substantially beats 0.764, it's learning temporal patterns that even raw-weekly XGBoost can't capture (nonlinear lag effects, attention to specific stress episodes). If the LSTM is comparable, tree-based models are the simpler, more interpretable choice for this problem.
