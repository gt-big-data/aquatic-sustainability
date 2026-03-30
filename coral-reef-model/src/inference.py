"""Sprint 3 Part C: Post-2020 inference (DHW response curve) and AIMS coral cover validation."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import TARGET_COL, YEAR_COL, GBR_COL, METADATA_COLS, FIGURES_DIR, PROJECT_ROOT


def build_reef_roster():
    """C.1: Build GBR reef roster from AIMS manta-tow data."""
    aims_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'aims', 'manta-tow-by-reef.csv')
    aims = pd.read_csv(aims_path)
    reef_roster = aims[['REEF_NAME', 'REEF_ID', 'LATITUDE', 'LONGITUDE', 'SECTOR']].drop_duplicates(subset='REEF_ID')
    print(f"\nC.1 — AIMS reef roster: {len(reef_roster)} unique reef sites")
    print(f"  Sectors: {reef_roster['SECTOR'].value_counts().to_dict()}")
    return reef_roster, aims


def plot_dhw_response_curve(best_model, scaler, train_df, feature_cols):
    """C.2: Synthetic DHW response curve — the most diagnostic plot."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    gbr_train = train_df[train_df[GBR_COL] == True]
    template = gbr_train[feature_cols].median()

    dhw_range = np.arange(0, 17, 0.5)
    predictions = []
    for dhw in dhw_range:
        row = template.copy()
        row['SSTA_DHW'] = dhw
        row['TSA_DHW'] = dhw
        row_scaled = scaler.transform(row.values.reshape(1, -1))
        pred = np.clip(best_model.predict(row_scaled)[0], 0, 1) * 100
        predictions.append(pred)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dhw_range, predictions, 'b-', linewidth=2)
    ax.axvline(4, color='orange', linestyle='--', label='Significant bleaching threshold (4 DHW)')
    ax.axvline(8, color='red', linestyle='--', label='Widespread mortality threshold (8 DHW)')
    ax.set_xlabel('Degree Heating Weeks (°C-weeks)')
    ax.set_ylabel('Predicted Bleaching %')
    ax.set_title('Model Bleaching Response to DHW (GBR median conditions)')
    ax.legend()
    ax.set_xlim(0, 16)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'dhw_response_curve.png')
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"\nC.2 — DHW response curve saved: {path}")
    print(f"  Predicted bleaching at 4 DHW: {predictions[8]:.1f}%")
    print(f"  Predicted bleaching at 8 DHW: {predictions[16]:.1f}%")
    print(f"  Predicted bleaching at 12 DHW: {predictions[24]:.1f}%")
    print(f"  Predicted bleaching at 16 DHW: {predictions[-1]:.1f}%")

    return dhw_range, predictions


def aims_coral_cover_analysis(aims):
    """C.3: AIMS coral cover year-over-year change analysis."""
    # Coerce MEAN_LIVE_CORAL to numeric
    aims = aims.copy()
    aims['MEAN_LIVE_CORAL'] = pd.to_numeric(aims['MEAN_LIVE_CORAL'], errors='coerce')
    aims['MEAN_DEAD_CORAL'] = pd.to_numeric(aims['MEAN_DEAD_CORAL'], errors='coerce')

    aims_sorted = aims.sort_values(['REEF_ID', 'REPORT_YEAR'])
    aims_sorted['coral_change'] = aims_sorted.groupby('REEF_ID')['MEAN_LIVE_CORAL'].diff()

    # Focus on 2020–2023
    recent = aims_sorted[aims_sorted['REPORT_YEAR'] >= 2020].dropna(subset=['coral_change'])

    print(f"\nC.3 — AIMS coral cover analysis (2020-2023)")
    print(f"  Total records with year-over-year change: {len(recent)}")

    if len(recent) > 0:
        biggest_losers = recent.nsmallest(20, 'coral_change')[
            ['REEF_NAME', 'REPORT_YEAR', 'MEAN_LIVE_CORAL', 'coral_change', 'LATITUDE', 'LONGITUDE']
        ]
        print(f"\n  Top 20 reefs with largest coral cover DECLINE (2020-2023):")
        print(biggest_losers.to_string(index=False))

        # Summary stats
        decline_count = (recent['coral_change'] < 0).sum()
        total = len(recent)
        mean_change = recent['coral_change'].mean()
        print(f"\n  Reefs with coral decline: {decline_count}/{total} ({decline_count/total*100:.0f}%)")
        print(f"  Mean coral cover change: {mean_change:.1f}%")

        # By sector
        print(f"\n  Mean coral cover change by sector:")
        sector_summary = recent.groupby('SECTOR')['coral_change'].agg(['mean', 'count'])
        for sector, row in sector_summary.iterrows():
            print(f"    {sector}: {row['mean']:.1f}% (n={int(row['count'])})")

        return biggest_losers, recent
    else:
        print("  No recent coral change data available.")
        return None, recent
