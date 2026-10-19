import os
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Settings
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "synthetic", "weekly_ts.csv"
)
ANOMALY_START = 300  # index of the first anomalous datapoint
ANOMALY_TREND = 0.01  # new trend after the anomaly (units per datapoint)
WARMUP = 52 * 4  # 4 years of weekly data: no detection during warm-up
CONFIDENCE = 0.95  # confidence level used for the time-series plot
CONFIDENCE_GRID = np.concatenate(
    [np.linspace(0.0, 0.99, 100), 1 - np.logspace(-2.5, -8, 30)]
)  # confidence levels swept to build the ROC and PR curves
ALARM_PROB_GRID = np.linspace(0.0, 1.0, 101)  # alarm probabilities, random detector
NUM_RANDOM_RUNS = 200  # Monte Carlo repetitions of the random detector
SEED = 0

# Load
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
y = df["y"].to_numpy()

# Trend anomaly: trend changes from 0 to ANOMALY_TREND at ANOMALY_START
steps_since_anomaly = np.maximum(np.arange(len(y)) - ANOMALY_START + 1, 0)
y_anomaly = y + ANOMALY_TREND * steps_since_anomaly

df["y_anomaly"] = y_anomaly
df["anomaly"] = (np.arange(len(y)) >= ANOMALY_START).astype(int)


# Dummy detector: fit a Gaussian to y_{1:t-1}, alarm if y_t is outside its
# `confidence` interval
def gaussian_point_detector(y, confidence=0.95, min_history=2):
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    alarms = np.zeros(len(y), dtype=int)
    for t in range(min_history, len(y)):
        mu = np.mean(y[:t])
        sigma = np.std(y[:t], ddof=1)
        if abs(y[t] - mu) > z * sigma:
            alarms[t] = 1
    return alarms


def confusion(labels, alarms):
    tp = np.sum((alarms == 1) & (labels == 1))
    fp = np.sum((alarms == 1) & (labels == 0))
    fn = np.sum((alarms == 0) & (labels == 1))
    tn = np.sum((alarms == 0) & (labels == 0))
    return tp, fp, fn, tn


def trapezoid(x, y):
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


# Random detector: alarm with probability alarm_prob, independent of the data
# and of the labels
def random_detector(n, alarm_prob, rng):
    return (rng.random(n) < alarm_prob).astype(int)


def roc_pr_curves(labels, alarm_sets):
    """ROC and PR curves (and their AUCs) from one alarm array per threshold."""
    fpr, tpr, precision, recall = [], [], [], []
    for alarms in alarm_sets:
        tp, fp, fn, tn = confusion(labels, alarms)
        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))
        recall.append(tp / (tp + fn))
        precision.append(tp / (tp + fp) if tp + fp > 0 else 1.0)

    # ROC curve: sort by FPR and anchor at (0, 0) and (1, 1)
    fpr, tpr = np.array(fpr), np.array(tpr)
    order = np.lexsort((tpr, fpr))
    fpr = np.concatenate([[0.0], fpr[order], [1.0]])
    tpr = np.concatenate([[0.0], tpr[order], [1.0]])

    # PR curve: sort by recall and anchor at recall = 0 with precision = 1
    recall, precision = np.array(recall), np.array(precision)
    order = np.lexsort((-precision, recall))
    recall = np.concatenate([[0.0], recall[order]])
    precision = np.concatenate([[1.0], precision[order]])

    return fpr, tpr, recall, precision, trapezoid(fpr, tpr), trapezoid(recall, precision)


df["alarm"] = gaussian_point_detector(
    y_anomaly, confidence=CONFIDENCE, min_history=WARMUP
)
alarm_idx = np.flatnonzero(df["alarm"])

# Sweep confidence levels; evaluate only after the warm-up
labels = df["anomaly"].to_numpy()[WARMUP:]
gaussian_alarms = [
    gaussian_point_detector(y_anomaly, confidence=conf, min_history=WARMUP)[WARMUP:]
    for conf in CONFIDENCE_GRID
]
fpr, tpr, recall, precision, auc_roc, auc_pr = roc_pr_curves(labels, gaussian_alarms)

# Random baseline: Monte Carlo runs of the random detector, scored the same way
rng = np.random.default_rng(SEED)
random_auc_roc, random_auc_pr = [], []
for run in range(NUM_RANDOM_RUNS):
    random_alarms = [random_detector(len(labels), p, rng) for p in ALARM_PROB_GRID]
    r_fpr, r_tpr, r_recall, r_precision, r_auc_roc, r_auc_pr = roc_pr_curves(
        labels, random_alarms
    )
    random_auc_roc.append(r_auc_roc)
    random_auc_pr.append(r_auc_pr)
    if run == 0:  # keep one run's curves for the plot
        rand_curves = (r_fpr, r_tpr, r_recall, r_precision)
random_auc_roc, random_auc_pr = np.array(random_auc_roc), np.array(random_auc_pr)

print(f"Evaluated points (after warm-up): {len(labels)}")
print(f"Abnormal points: {labels.sum()} ({labels.mean():.1%})")
print(f"{'':<18}{'AUC-ROC':>16}{'AUC-PR':>16}")
print(f"{'Gaussian detector':<18}{auc_roc:>16.4f}{auc_pr:>16.4f}")
print(
    f"{'Random detector':<18}"
    f"{f'{random_auc_roc.mean():.4f} ± {random_auc_roc.std():.4f}':>16}"
    f"{f'{random_auc_pr.mean():.4f} ± {random_auc_pr.std():.4f}':>16}"
)
print(f"(random: mean ± std over {NUM_RANDOM_RUNS} runs)")

# Plot: time series with alarms at CONFIDENCE
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["date"], df["y"], lw=1, color="gray", alpha=0.6, label="original")
ax.plot(df["date"], df["y_anomaly"], lw=1, color="C0", label="with trend anomaly")
ax.axvspan(
    df["date"].iloc[0], df["date"].iloc[WARMUP], color="gray", alpha=0.15, label="warm-up"
)
ax.axvline(
    df["date"].iloc[ANOMALY_START], color="red", ls="--", lw=1, label="anomaly start"
)
ax.scatter(
    df["date"].iloc[alarm_idx],
    df["y_anomaly"].iloc[alarm_idx],
    s=12,
    color="red",
    zorder=3,
    label=f"alarm ({CONFIDENCE:.0%})",
)
ax.set_xlabel("Date")
ax.set_ylabel("y")
ax.set_title(f"Trend anomaly: trend 0 → {ANOMALY_TREND} per datapoint")
ax.legend()
fig.tight_layout()

# Plot: ROC and PR curves
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4.5))
r_fpr, r_tpr, r_recall, r_precision = rand_curves
ax_roc.plot(fpr, tpr, color="C0", lw=1.5, label="Gaussian detector")
ax_roc.plot(r_fpr, r_tpr, color="gray", ls="--", lw=1, label="random detector")
ax_roc.set_xlabel("False positive rate")
ax_roc.set_ylabel("True positive rate")
ax_roc.set_title(f"ROC curve (AUC = {auc_roc:.3f})")
ax_roc.set_xlim(0, 1)
ax_roc.set_ylim(0, 1.02)
ax_roc.legend(loc="lower right")

ax_pr.plot(recall, precision, color="C1", lw=1.5, label="Gaussian detector")
ax_pr.plot(r_recall, r_precision, color="gray", ls="--", lw=1, label="random detector")
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.set_title(f"PR curve (AUC = {auc_pr:.3f})")
ax_pr.set_xlim(0, 1)
ax_pr.set_ylim(0, 1.02)
ax_pr.legend(loc="lower left")
fig.tight_layout()

plt.show()
