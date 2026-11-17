"""Dummy config: apply single anomalies of type ANM_TYPE to the stationary synthetic series.

Loads the stationary series from data/synthetic/syn_values.csv and the anomaly info from
data/anomaly_info/anm_single_location.csv, then (see src/tsbenchmark/build_dataset.py):
    - keeps the first NUM_REP anm_start_ratio values within ANM_START_RANGE
    - for each magnitude in ANM_MAG_LIST, with both signs, and each kept start ratio,
      applies one anomaly of type ANM_TYPE from the start point (e.g. "trend": the zero
      trend becomes anm_mag, in units per year)
    - if PLOT_DATASET is True, plots each stationary series with the anomalies
      applied on top

Usage:
    python -m examples.config_dummy
"""

import os

from src.tsbenchmark.build_dataset import build_dataset

# Inputs
# Anomaly type: "no_anomaly" (stationary series, unchanged), "trend", "level",
# "global_outlier", "contextual_outlier", "noise_range", "scale_range"
DATA_SAMPLING_FREQ = "weekly"  # "daily", "weekly", "monthly" or "quarterly"

# # Stationary time series 
# ANM_TYPE = "no_anomaly"
# ANM_MAG_LIST = None
# ANM_START_RANGE = (0.3, 0.6)

# Time series with regime anomalies
ANM_TYPE = "trend"
ANM_MAG_LIST = [0.2, 0.4, 0.6, 0.8, 1.0]
ANM_START_RANGE = (0.3, 0.6)  # allowed anm_start_ratio

# # Time series with point anomalies
# ANM_TYPE = "global_outlier"
# ANM_MAG_LIST = [0.05, 0.1, 0.2, 0.3, 0.5] # Global outlier: |mag| = how far beyond the series' max/min, in std of the series
# # ANM_TYPE = "contextual_outlier"
# # ANM_MAG_LIST = [0.5, 0.75, 1.0] # Contextual outlier: |mag| = how far from the series' own value, in std of the series
# ANM_START_RANGE = (0.3, 0.8)  # allowed anm_start_ratio

# # Time series with range anomalies (each lasts three years)
# # ANM_TYPE = "noise_range"
# # ANM_MAG_LIST = [0.2, 0.4, 0.6, 0.8, 1.0]  # std of the added noise, in std of the series
# ANM_TYPE = "scale_range"
# ANM_MAG_LIST = [0.5, 1.0, 1.5, 2.0]  # change of the added noise or yearly cycle's amplitude, in std of the series;
# ANM_START_RANGE = (0.3, 0.6)  # allowed anm_start_ratio

NUM_REP = 20  # number of anm_start_ratio values to use
PLOT_DATASET = True  # plot each stationary series with its anomalies on top

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TS_PATH = os.path.join(DATA_DIR, "synthetic", "syn_values.csv")
ANM_LOC_PATH = os.path.join(DATA_DIR, "anomaly_info", "anm_single_location.csv")

# Load the stationary series and apply every anomaly of type ANM_TYPE:
# series_with_anomaly[name][(anm_mag, rep)] is one series with an anomaly applied
# anm_labels[name][(anm_mag, rep)]: 1 at the anomalous time steps of that series, 0 elsewhere
stationary_series, series_with_anomaly, anm_labels, anm_start_ratio, anm_mags = build_dataset(
    TS_PATH,
    ANM_LOC_PATH,
    anm_type=ANM_TYPE,
    anm_mag_list=ANM_MAG_LIST,
    anm_start_range=ANM_START_RANGE,
    num_rep=NUM_REP,
    data_sampling_freq=DATA_SAMPLING_FREQ,
    plot_dataset=PLOT_DATASET,
)
