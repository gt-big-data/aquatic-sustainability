import os

# Column name constants (actual GCBD CSV names, NOT PRD assumptions)
TARGET_COL = 'Percent_Bleaching'
YEAR_COL = 'Date_Year'
LAT_COL = 'Latitude_Degrees'
LON_COL = 'Longitude_Degrees'
SECTOR_COL = 'Sector'
GBR_COL = 'is_GBR'

# GBR bounding box
GBR_BOUNDS = {
    'lat_min': -25.0,
    'lat_max': -10.0,
    'lon_min': 142.0,
    'lon_max': 154.0,
}

# Metadata columns for splitting/evaluation — NOT model features
METADATA_COLS = [YEAR_COL, GBR_COL, SECTOR_COL]

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_GCBD_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'gcbd', 'global_bleaching_environmental.csv')
PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'global_training_data.csv')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'figures')

# 32 CoRTAD v6 thermal feature columns
CORTAD_COLS = [
    'ClimSST',
    'Temperature_Kelvin', 'Temperature_Mean', 'Temperature_Minimum',
    'Temperature_Maximum', 'Temperature_Kelvin_Standard_Deviation',
    'SSTA', 'SSTA_Standard_Deviation', 'SSTA_Mean', 'SSTA_Minimum', 'SSTA_Maximum',
    'SSTA_Frequency', 'SSTA_Frequency_Standard_Deviation', 'SSTA_FrequencyMax', 'SSTA_FrequencyMean',
    'SSTA_DHW', 'SSTA_DHW_Standard_Deviation', 'SSTA_DHWMax', 'SSTA_DHWMean',
    'TSA', 'TSA_Standard_Deviation', 'TSA_Minimum', 'TSA_Maximum', 'TSA_Mean',
    'TSA_Frequency', 'TSA_Frequency_Standard_Deviation', 'TSA_FrequencyMax', 'TSA_FrequencyMean',
    'TSA_DHW', 'TSA_DHW_Standard_Deviation', 'TSA_DHWMax', 'TSA_DHWMean',
]

# Site metadata numeric columns
SITE_NUMERIC_COLS = ['Distance_to_Shore', 'Turbidity', 'Cyclone_Frequency', 'Windspeed']

# Geographic columns (now features, not just metadata)
GEO_FEATURE_COLS = [LAT_COL, LON_COL]

# All numeric feature columns (before one-hot encoding)
NUMERIC_FEATURE_COLS = CORTAD_COLS + SITE_NUMERIC_COLS + GEO_FEATURE_COLS

# Columns to drop (>30% NaN)
DROP_COLS = ['Depth_m', 'Percent_Cover']


def assign_sector(lat):
    if lat >= -16.0:
        return 'Northern'
    elif lat >= -20.0:
        return 'Central'
    else:
        return 'Southern'
