"""Dummy config: apply anomalies of type ANM_TYPE to stationary series (see
src/tsbenchmark/build_dataset.py).

Usage:
    python -m examples.config_dummy
"""

import os

from tsbenchmark.build_dataset import build_dataset
from tsbenchmark.detector.online.gaussian_naive_online_detector import GaussianNaiveOnlineDetector
from tsbenchmark.detector.offline.gaussian_naive_offline_detector import GaussianNaiveOfflineDetector
from tsbenchmark.evaluation_metrics.online.regime_online_metrics import (
    detection_probability, false_alarm_rate, repeated_alarm_rate, time_to_detection, vus_amoc
)
from tsbenchmark.evaluation_metrics.point_metrics import (
    auc_pr, auc_roc, f1, fn, fp, precision, recall, tn, tp
)
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
detector = GaussianNaiveOnlineDetector(threshold=THRESHOLD)
# detector = GaussianNaiveOfflineDetector(threshold=THRESHOLD)
# NOTE: anm_labels_pred only covers the steps after the training part (no prediction there), so
# it is train_len steps shorter than anm_labels: anm_labels_pred[name][key][i] is the prediction
# of step train_len + i, to compare with anm_labels[name][key][train_len:].
# Run on every series with anomaly: each name and each of its (anm_mag, rep) keys. The detector is
# refitted on the training part of each series, then detects on the rest.
anm_labels_pred, anm_labels_test, anm_scores_pred = {}, {}, {}
for name in series_with_anomaly:
    anm_labels_pred[name], anm_labels_test[name], anm_scores_pred[name] = {}, {}, {}
    for key, y in series_with_anomaly[name].items():
        anm_labels_pred[name][key] = detector.fit(y[:train_len]).detect(y[train_len:])
        anm_scores_pred[name][key] = detector.scores  # scores kept by detect, for VUS-AMOC
        anm_labels_test[name][key] = anm_labels[name][key][train_len:]

# Regime online metrics over all series with anomaly
keys = [(name, key) for name in anm_labels_test for key in anm_labels_test[name]]
labels_test = [anm_labels_test[n][k] for n, k in keys]
labels_pred = [anm_labels_pred[n][k] for n, k in keys]
mags = [abs(k[0]) for _, k in keys]  # absolute magnitudes of ANM_MAG_LIST
print(f"Regime online metrics over {len(keys)} series:")
print(
    "false_alarm_rate (per 10 years)",
    false_alarm_rate(labels_test, labels_pred, DATA_SAMPLING_FREQ),
)
print(
    "repeated_alarm_rate (per 10 years)",
    repeated_alarm_rate(labels_test, labels_pred, DATA_SAMPLING_FREQ),
)
print(
    "time_to_detection (years) per magnitude",
    time_to_detection(labels_test, labels_pred, mags, DATA_SAMPLING_FREQ, plot=True),
)
print(
    "detection_probability per magnitude",
    detection_probability(labels_test, labels_pred, mags, DATA_SAMPLING_FREQ, plot=True),
)

# VUS-AMOC sweeps the threshold over the scores: THRESHOLD above does not change it
scores_pred = [anm_scores_pred[n][k] for n, k in keys]
print("vus_amoc", vus_amoc(labels_test, scores_pred, mags, DATA_SAMPLING_FREQ, plot=True))
