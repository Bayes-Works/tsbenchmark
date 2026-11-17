"""Dummy config: apply single anomalies of type ANM_TYPE to the stationary synthetic series.

Loads the stationary series from data/synthetic/syn_values.csv and the anomaly info from
data/anomaly_info/anm_single_location.csv, then (see src/tsbenchmark/build_dataset.py):
    - keeps the first NUM_REP anm_start_ratio values within ANM_START_RANGE
    - for each magnitude in ANM_MAG_LIST, with both signs, and each kept start ratio,
      applies one anomaly of type ANM_TYPE from the start point (e.g. "trend": the zero
      trend becomes anm_mag, in units per NUM_POINTS_PER_YEAR data points)
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
ANM_TYPE = "no_anomaly"
# Absolute anomaly magnitudes to test, in the standardized space of each stationary
# series (1.0 = one std of that series; for "trend", per NUM_POINTS_PER_YEAR data
# points). Scaled to each series' units before the anomaly is applied. Any spacing.
# Each is applied with both signs: -1.0, ..., -0.2, 0.2, ..., 1.0
# ANM_MAG_LIST = [0.2, 0.4, 0.6, 0.8, 1.0]
ANM_START_RANGE = (0.3, 0.6)  # allowed anm_start_ratio
NUM_REP = 20  # number of anm_start_ratio values to use
NUM_POINTS_PER_YEAR = 52
PLOT_DATASET = True  # plot each stationary series with its anomalies on top

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TS_PATH = os.path.join(DATA_DIR, "synthetic", "syn_values.csv")
ANM_PATH = os.path.join(DATA_DIR, "anomaly_info", "anm_single_location.csv")

# Load the stationary series and apply every anomaly of type ANM_TYPE:
# series_with_anomaly[name][(anm_mag, rep)] is one series with an anomaly applied
stationary_series, series_with_anomaly, anm_start_ratio, anm_mags = build_dataset(
    TS_PATH,
    ANM_PATH,
    anm_type=ANM_TYPE,
    # anm_mag_list=ANM_MAG_LIST,
    anm_start_range=ANM_START_RANGE,
    num_rep=NUM_REP,
    num_points_per_year=NUM_POINTS_PER_YEAR,
    plot_dataset=PLOT_DATASET,
)
