"""Sprint 1 pipeline: data ingestion, cleaning, feature assembly, and EDA."""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingest import load_gcbd
from src.features import prepare_features
from src.preprocess import scale_target, tag_regions
from src.utils import (
    TARGET_COL, YEAR_COL, LAT_COL, LON_COL, SECTOR_COL, GBR_COL,
    METADATA_COLS, PROCESSED_PATH, FIGURES_DIR, GBR_BOUNDS,
)


def save_processed(df, feature_cols):
    """Select final columns, verify zero NaN, save to CSV."""
    # Feature cols + target + metadata
    all_cols = [TARGET_COL] + feature_cols + METADATA_COLS
    # Ensure all exist
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df[all_cols].copy()

    total_nan = out.isna().sum().sum()
    assert total_nan == 0, f"Expected zero NaN but found {total_nan}"

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    out.to_csv(PROCESSED_PATH, index=False)

    # Summary
    gbr = out[out[GBR_COL]]
    print(f"\n{'='*60}")
    print(f"SAVED: {PROCESSED_PATH}")
    print(f"{'='*60}")
    print(f"Total rows:   {len(out)}")
    print(f"GBR rows:     {len(gbr)}")
    print(f"Non-GBR rows: {len(out) - len(gbr)}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")
    print(f"Year range: {int(out[YEAR_COL].min())}–{int(out[YEAR_COL].max())}")
    print(f"\nGBR sector distribution:")
    print(gbr[SECTOR_COL].value_counts().to_string())
    print(f"\nTarget (global): mean={out[TARGET_COL].mean():.4f}, median={out[TARGET_COL].median():.4f}, std={out[TARGET_COL].std():.4f}")
    print(f"Target (GBR):    mean={gbr[TARGET_COL].mean():.4f}, median={gbr[TARGET_COL].median():.4f}, std={gbr[TARGET_COL].std():.4f}")


def run_eda(df, feature_cols):
    """EDA sanity checks and plots."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"EDA SANITY CHECKS")
    print(f"{'='*60}")

    # --- 1. Top-10 correlations with target (global) ---
    numeric_feats = [c for c in feature_cols if df[c].dtype in [np.float64, np.int64, float, int]]
    corr_global = df[numeric_feats + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False)
    print(f"\nTop-10 features correlated with target (GLOBAL):")
    for feat, val in corr_global.head(10).items():
        print(f"  {feat}: {val:.4f}")

    # --- 2. Top-10 correlations (GBR only) ---
    gbr = df[df[GBR_COL]]
    corr_gbr = gbr[numeric_feats + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False)
    print(f"\nTop-10 features correlated with target (GBR ONLY):")
    for feat, val in corr_gbr.head(10).items():
        print(f"  {feat}: {val:.4f}")

    # Save top-10 bar plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    corr_global.head(10).plot.barh(ax=axes[0], color='steelblue')
    axes[0].set_title('Top-10 Feature Correlations (Global)')
    axes[0].set_xlabel('|Pearson r|')
    axes[0].invert_yaxis()
    corr_gbr.head(10).plot.barh(ax=axes[1], color='coral')
    axes[1].set_title('Top-10 Feature Correlations (GBR)')
    axes[1].set_xlabel('|Pearson r|')
    axes[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'correlation_top10.png'), dpi=150)
    plt.close(fig)
    print(f"\nSaved: correlation_top10.png")

    # --- 3. Collinearity check (CoRTAD columns) ---
    from src.utils import CORTAD_COLS
    cortad_present = [c for c in CORTAD_COLS if c in df.columns]
    corr_matrix = df[cortad_present].corr().abs()
    # Get upper triangle pairs > 0.95
    high_corr_pairs = []
    for i in range(len(cortad_present)):
        for j in range(i + 1, len(cortad_present)):
            val = corr_matrix.iloc[i, j]
            if val > 0.95:
                high_corr_pairs.append((cortad_present[i], cortad_present[j], val))
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"\nCoRTAD collinearity pairs (|r| > 0.95): {len(high_corr_pairs)}")
    for a, b, v in high_corr_pairs:
        print(f"  {a} <-> {b}: {v:.4f}")

    # --- 4. Target by year (GBR) ---
    gbr_by_year = gbr.groupby(YEAR_COL)[TARGET_COL].agg(['mean', 'count'])
    print(f"\nMean bleaching by year (GBR):")
    for year, row in gbr_by_year.iterrows():
        marker = " <<<" if row['mean'] > 0.25 else ""
        print(f"  {int(year)}: mean={row['mean']:.4f} (n={int(row['count'])}){marker}")

    # --- 5. Target by sector (GBR) ---
    gbr_by_sector = gbr.groupby(SECTOR_COL)[TARGET_COL].agg(['mean', 'count'])
    print(f"\nMean bleaching by sector (GBR):")
    for sector, row in gbr_by_sector.iterrows():
        print(f"  {sector}: mean={row['mean']:.4f} (n={int(row['count'])})")

    # --- 6. Global geographic spread ---
    if 'Ocean_Name' in df.columns:
        print(f"\nGlobal coverage by ocean:")
        print(df['Ocean_Name'].value_counts().head(10).to_string())
    if 'Country_Name' in df.columns:
        n_countries = df['Country_Name'].nunique()
        print(f"\nUnique countries: {n_countries}")
    if 'Ecoregion_Name' in df.columns:
        n_eco = df['Ecoregion_Name'].nunique()
        print(f"Unique ecoregions: {n_eco}")

    # --- 7. Global coverage scatter plot ---
    fig, ax = plt.subplots(figsize=(16, 8))
    scatter = ax.scatter(
        df[LON_COL], df[LAT_COL],
        c=df[TARGET_COL], cmap='YlOrRd', s=3, alpha=0.5,
        vmin=0, vmax=1,
    )
    plt.colorbar(scatter, ax=ax, label='Bleaching (0-1)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Global Training Data Coverage (n={len(df):,})')
    # Highlight GBR box
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (GBR_BOUNDS['lon_min'], GBR_BOUNDS['lat_min']),
        GBR_BOUNDS['lon_max'] - GBR_BOUNDS['lon_min'],
        GBR_BOUNDS['lat_max'] - GBR_BOUNDS['lat_min'],
        linewidth=2, edgecolor='blue', facecolor='none', linestyle='--', label='GBR'
    )
    ax.add_patch(rect)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'global_coverage.png'), dpi=150)
    plt.close(fig)
    print(f"\nSaved: global_coverage.png")

    # --- Target distribution histogram ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df[TARGET_COL], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title(f'Target Distribution (Global, n={len(df):,})')
    axes[0].set_xlabel('Percent Bleaching (scaled)')
    axes[0].set_ylabel('Count')
    axes[1].hist(gbr[TARGET_COL], bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[1].set_title(f'Target Distribution (GBR, n={len(gbr):,})')
    axes[1].set_xlabel('Percent Bleaching (scaled)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'target_distribution.png'), dpi=150)
    plt.close(fig)
    print(f"Saved: target_distribution.png")


def run_sprint1():
    print("=" * 60)
    print("SPRINT 1: Data Ingestion, Cleaning, Feature Assembly")
    print("=" * 60)

    df = load_gcbd()

    # Keep some metadata columns around for EDA (Ocean_Name, Country_Name, Ecoregion_Name)
    extra_meta = [c for c in ['Ocean_Name', 'Country_Name', 'Ecoregion_Name'] if c in df.columns]

    df_full = df.copy()  # keep original for EDA geographic columns

    df, feature_cols = prepare_features(df)
    df = scale_target(df)
    df = tag_regions(df)
    save_processed(df, feature_cols)

    # For EDA, re-attach geographic metadata from original
    if extra_meta:
        df_eda = df.copy()
        for col in extra_meta:
            df_eda[col] = df_full.loc[df.index, col].values if len(df_full.loc[df.index]) == len(df) else None
        run_eda(df_eda, feature_cols)
    else:
        run_eda(df, feature_cols)

    print(f"\n{'='*60}")
    print("SPRINT 1 COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    run_sprint1()
