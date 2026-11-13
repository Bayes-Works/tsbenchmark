"""Dummy config: apply single trend anomalies to the stationary synthetic series.

Loads the stationary series from data/synthetic/syn_values.csv and the anomaly info from
data/anomaly_info/anm_single_location.csv, then:
    - keeps the first NUM_REP anm_start_ratio values within ANM_START_RANGE
    - for each anm_mag in the magnitudes and each kept start ratio, applies a trend
      anomaly: from the start point onward the zero trend becomes anm_mag
      (in units per NUM_POINTS_PER_CYCLE data points)
    - plots each stationary series with the anomalies applied on top

Usage:
    python examples/config_dummy.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Inputs
ANM_MAG_RANGE = (0.1, 1.0)  # anomaly magnitude, unit / NUM_POINTS_PER_CYCLE data points
ANM_MAG_STEP = 0.1  # -> magnitudes 0.1, 0.2, ..., 1.0
ANM_START_RANGE = (0.3, 0.6)  # allowed anm_start_ratio
NUM_REP = 20  # number of anm_start_ratio values to use
NUM_POINTS_PER_CYCLE = 52

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TS_PATH = os.path.join(DATA_DIR, "synthetic", "syn_values.csv")
ANM_PATH = os.path.join(DATA_DIR, "anomaly_info", "anm_single_location.csv")

# Load the stationary time series and the anomaly info
# Wide CSV, one column per series (ts1, ts2, ...); shorter series are padded with
# empty cells at the bottom, so drop the NaNs to recover each series' own length.
values = pd.read_csv(TS_PATH)
series = {name: values[name].dropna().to_numpy() for name in values.columns}
anm_info = pd.read_csv(ANM_PATH)  # column: anm_start_ratio

# First NUM_REP anm_start_ratio values within ANM_START_RANGE
anm_start_ratio = [
    ratio
    for ratio in anm_info["anm_start_ratio"]
    if ANM_START_RANGE[0] <= ratio <= ANM_START_RANGE[1]
][:NUM_REP]

anm_mags = np.round(
    np.arange(ANM_MAG_RANGE[0], ANM_MAG_RANGE[1] + 1e-9, ANM_MAG_STEP), 2
)
# Mirror to the negative side: -1.0, ..., -0.1, 0.1, ..., 1.0
anm_mags = np.concatenate([-anm_mags[::-1], anm_mags])


def apply_trend_anomaly(ts, start_ratio, mag):
    """From start_ratio onward, change the zero trend to a slope of mag
    (unit / NUM_POINTS_PER_CYCLE data points)."""
    y = np.asarray(ts, dtype=float).copy()
    t_start = int(round(start_ratio * (len(y) - 1)))
    y[t_start:] += mag / NUM_POINTS_PER_CYCLE * np.arange(len(y) - t_start)
    return y

# Apply: one anomalous series per (series, anm_mag, rep)
anomalous = {
    name: {
        (mag, rep): apply_trend_anomaly(ts, anm_start_ratio[rep], mag)
        for mag in anm_mags
        for rep in range(len(anm_start_ratio))
    }
    for name, ts in series.items()
}

# Visualize: one subplot per stationary series, with ALL anomalies on top
# (every rep at every magnitude), colored by absolute magnitude so the negative
# side mirrors the positive side's colors.
fig, axes = plt.subplots(5, 4, figsize=(16, 8), sharex=True)
cmap = plt.get_cmap("viridis")
abs_min, abs_max = np.abs(anm_mags).min(), np.abs(anm_mags).max()
for i, (name, ax) in enumerate(zip(series, axes.ravel())):
    for rep in range(len(anm_start_ratio)):
        for mag in anm_mags:
            ax.plot(
                anomalous[name][(mag, rep)],
                lw=0.5,
                alpha=0.25,
                color=cmap((abs(mag) - abs_min) / (abs_max - abs_min)),
            )
    ax.plot(series[name], lw=1.8, color="black")
    ax.set_title(
        f"{name}: {len(anm_start_ratio)} reps x {len(anm_mags)} mags "
        f"({anm_mags[0]}..{anm_mags[-1]})",
        fontsize=8,
    )
fig.suptitle("Stationary series (black) with all trend anomalies (color = |magnitude|)")
fig.tight_layout()
plt.show()
