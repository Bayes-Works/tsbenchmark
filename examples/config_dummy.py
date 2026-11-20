"""Dummy config: apply anomalies of type ANM_TYPE to stationary series (see
src/tsbenchmark/build_dataset.py).

Usage:
    python -m examples.config_dummy
"""

import os

from tsbenchmark.build_dataset import build_dataset
from tsbenchmark.detector.online.gaussian_naive_online_detector import GaussianNaiveOnlineDetector
from tsbenchmark.detector.offline.gaussian_naive_offline_detector import GaussianNaiveOfflineDetector
from tsbenchmark.utils import get_num_points_per_year

# Inputs
DATA_SAMPLING_FREQ = "weekly"  # "daily", "weekly", "monthly" or "quarterly"

# ANM_TYPE and what each magnitude in ANM_MAG_LIST means (in std of the series):
#   "no_anomaly"          stationary series, unchanged (ANM_MAG_LIST = None)
#   "trend"               slope per year, from the start to the end of the series
#   "level"               level shift, from the start to the end of the series
#   "global_outlier"      distance beyond the series' max or min, at one step
#   "contextual_outlier"  deviation at one step, kept inside the series' [min, max]
#   "noise_range"         std of Gaussian noise added over three years
#   "scale_range"         change of the yearly cycle's amplitude over three years
ANM_TYPE = "trend"
ANM_MAG_LIST = [0.2, 0.4, 0.6, 0.8, 1.0]  # absolute values, mirrored to both signs (not noise)
ANM_START_RANGE = (0.3, 0.6)  # allowed anm_start_ratio

NUM_REP = 20  # use the first NUM_REP anm_start_ratio values within ANM_START_RANGE
NUM_SERIES = None  # load the first NUM_SERIES columns of TS_PATH; None = all
PLOT_DATASET = True

THRESHOLD = 0.99  # alarm when y_t is outside the central 99% interval of the Gaussian of the past
NUM_TRAIN_CYCLES = 3  # the first NUM_TRAIN_CYCLES yearly cycles train the detector

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TS_PATH = os.path.join(DATA_DIR, "synthetic", "syn_values.csv")
# TS_PATH = os.path.join(DATA_DIR, "real100", "weekly_values.csv")
ANM_LOC_PATH = os.path.join(DATA_DIR, "anomaly_info", "anm_single_location.csv")

stationary_series, series_with_anomaly, anm_labels, anm_start_ratio, anm_mags = build_dataset(
    TS_PATH,
    ANM_LOC_PATH,
    anm_type=ANM_TYPE,
    anm_mag_list=ANM_MAG_LIST,
    anm_start_range=ANM_START_RANGE,
    num_rep=NUM_REP,
    data_sampling_freq=DATA_SAMPLING_FREQ,
    num_series=NUM_SERIES,
    plot_dataset=PLOT_DATASET,
)

# Detector
train_len = NUM_TRAIN_CYCLES * get_num_points_per_year(DATA_SAMPLING_FREQ)
# detector = GaussianNaiveOnlineDetector(threshold=THRESHOLD)
detector = GaussianNaiveOfflineDetector(threshold=THRESHOLD)
# NOTE: anm_labels_pred only covers the steps after the training part (no prediction there), so
# it is train_len steps shorter than anm_labels: anm_labels_pred[name][key][i] is the prediction
# of step train_len + i, to compare with anm_labels[name][key][train_len:].
# Run only on the first series: the first name and its first (anm_mag, rep) key.
name = next(iter(series_with_anomaly))
key, y = next(iter(series_with_anomaly[name].items()))
anm_labels_pred = {
    name: {
        key: detector.fit(y[:train_len]).detect(
            y[train_len:], plot=True, anm_labels=anm_labels[name][key]
        )
    }
}

# Next next step: evaluation metrics to compare the detector's detections with anm_labels