"""Constants and helpers for the AIMS coral cover prediction project."""

import os

# === Paths ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_AIMS_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'aims')
RAW_NOAA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'noaa-crw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'reports')

# === Column names ===
TARGET_COL = 'MEAN_LIVE_CORAL'
REEF_ID_COL = 'REEF_ID'
YEAR_COL = 'REPORT_YEAR'
SECTOR_COL = 'SECTOR'

# === Temporal split ===
TEMPORAL_CUTOFF = 2020  # train < 2020, test >= 2020

# === Feature columns (final model-ready dataset) ===
FEATURE_COLS = [
    'LIVE_CORAL_LAG1', 'DEAD_CORAL_LAG1', 'COTS_LAG1',
    'CORAL_TRAJECTORY', 'YEAR_GAP',
    'DHW_MAX', 'DHW_MAX_PREV', 'SSTA_MAX', 'SSTA_MAX_PREV',
    'LATITUDE', 'LONGITUDE',
    'SHELF_M', 'SHELF_O',
]

METADATA_COLS = [REEF_ID_COL, 'REEF_NAME', YEAR_COL, SECTOR_COL]

# === Thresholds ===
MAX_YEAR_GAP = 3  # max year gap for lagged features to be considered valid
