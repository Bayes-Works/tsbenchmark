"""Range anomaly benchmark: inject ANM_TYPE anomalies into stationary series, run
the Gaussian naive offline detector, save its outputs to OUTPUT_PATH, and report
evaluation metrics.

Usage:
    python -m examples.config_range
"""

import os

from tsbenchmark.build_dataset import build_dataset
from tsbenchmark.detector.offline.gaussian_naive_offline_detector import GaussianNaiveOfflineDetector
from tsbenchmark.evaluation_metrics.offline.range_offline_metrics import affiliation_f1
from tsbenchmark.utils import get_num_points_per_year, save_outputs

# Inputs
# Define anomalies and time series info
ANM_TYPE = "noise_range"
ANM_MAG_LIST = [0.2, 0.4, 0.6, 0.8, 1.0]  # absolute values, mirrored to both signs (not noise)
ANM_START_RANGE = (0.3, 0.6)  # allowed anm_start_ratio
NUM_REP = 20  # use the first NUM_REP anm_start_ratio values within ANM_START_RANGE
NUM_SERIES = None  # load the first NUM_SERIES columns of TS_PATH; None = all
PLOT_DATASET = True
THRESHOLD = 0.99  # alarm when y_t is outside the central 99% interval of the Gaussian of the past
NUM_TRAIN_CYCLES = 3  # the first NUM_TRAIN_CYCLES yearly cycles train the detector
DATA_SAMPLING_FREQ = "weekly"  # "daily", "weekly", "monthly" or "quarterly"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TS_PATH = os.path.join(DATA_DIR, "synthetic", "syn_values.csv")
# TS_PATH = os.path.join(DATA_DIR, "real100", "weekly_values.csv")
ANM_LOC_PATH = os.path.join(DATA_DIR, "anomaly_info", "anm_single_location.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "config_range_outputs.json")


# Load time series and anomalies
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
detector = GaussianNaiveOfflineDetector(threshold=THRESHOLD)
# Run on every series with anomaly.
anm_labels_pred, anm_labels_test, anm_scores_pred = {}, {}, {}
for name in series_with_anomaly:
    anm_labels_pred[name], anm_labels_test[name], anm_scores_pred[name] = {}, {}, {}
    for key, y in series_with_anomaly[name].items():
        anm_labels_pred[name][key] = detector.fit(y[:train_len]).detect(y[train_len:])
        anm_scores_pred[name][key] = detector.scores  # scores kept by detect, for VUS-AMOC
        anm_labels_test[name][key] = anm_labels[name][key][train_len:]


# Save results for comparison between different configs
save_outputs(OUTPUT_PATH, anm_labels_pred, anm_scores_pred, anm_labels_test)


# Evaluation
# Range offline metrics over all series with anomaly
keys = [(name, key) for name in anm_labels_test for key in anm_labels_test[name]]
labels_test = [anm_labels_test[n][k] for n, k in keys]
labels_pred = [anm_labels_pred[n][k] for n, k in keys]
mags = [abs(k[0]) for _, k in keys]  # absolute magnitudes of ANM_MAG_LIST
print(f"Range offline metrics over {len(keys)} series:")
print("affiliation_f1 per magnitude", affiliation_f1(labels_test, labels_pred, mags, plot=True))

