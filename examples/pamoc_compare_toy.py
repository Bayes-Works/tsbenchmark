# %% Load data and create time series with anomalies
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT_DIR, os.path.join(ROOT_DIR, "src")]
from src.tsbenchmark.detector.gaussian_point_detector import GaussianPointDetector

DATA_DIR = os.path.join(ROOT_DIR, "data")
MAGNITUDES = np.round(np.arange(1, 11) * 0.01, 2)  # normalized: 0.1, 0.2, ..., 1.0
NUM_TS_PER_MAG = 10
NUM_PLOT_PER_MAG = 3

#######################################################################################
############################### Load base series and generate anomalies ###############
#######################################################################################


# Base time series
base_df = pd.read_csv(os.path.join(DATA_DIR, "synthetic", "weekly_ts.csv"), parse_dates=["date"])
dates, y_base = base_df["date"], base_df["y"].to_numpy()
num_steps = len(y_base)

# Anomaly info: first NUM_TS_PER_MAG start ratios of each selected magnitude
with open(os.path.join(DATA_DIR, "anomaly_info", "single_anm_info.json")) as f:
    anm_info = pd.DataFrame(json.load(f))
anm_info = (
    anm_info[np.isin(anm_info["anm_mag_norm"].round(2), MAGNITUDES)]
    .groupby("anm_mag_norm", sort=True)
    .head(NUM_TS_PER_MAG)
    .reset_index(drop=True)
)

# Denormalize: a trend of m per step on (y - mean) / std is m * std per step on y
anm_info["anm_mag"] = anm_info["anm_mag_norm"] * y_base.std()
anm_info["anm_start"] = (anm_info["anm_start_ratio"] * num_steps).astype(int)

# Trend anomaly: trend changes from 0 to anm_mag at anm_start
steps = np.arange(num_steps)
steps_since_anm = np.maximum(steps[None, :] - anm_info["anm_start"].to_numpy()[:, None] + 1, 0)
ts_anm = y_base[None, :] + anm_info["anm_mag"].to_numpy()[:, None] * steps_since_anm
labels = (steps_since_anm > 0).astype(int)  # 1 from the anomaly start onward

print(f"Base series: {num_steps} steps, mean = {y_base.mean():.3f}, std = {y_base.std():.3f}")
print(f"Generated {len(ts_anm)} time series ({len(MAGNITUDES)} magnitudes x {NUM_TS_PER_MAG})")

# # Plot NUM_PLOT_PER_MAG examples per magnitude
# fig, axes = plt.subplots(2, 5, figsize=(20, 6), sharex=True)
# for ax, (mag_norm, group) in zip(axes.flat, anm_info.groupby("anm_mag_norm")):
#     ax.plot(dates, y_base, color="gray", lw=0.8, alpha=0.6)
#     for i, idx in enumerate(group.index[:NUM_PLOT_PER_MAG]):
#         start = group.at[idx, "anm_start"]
#         ax.plot(dates, ts_anm[idx], color=f"C{i}", lw=0.8)
#         ax.axvline(dates.iloc[start], color=f"C{i}", ls="--", lw=0.8)
#     ax.set_title(f"mag_norm = {mag_norm:.1f} (mag = {group['anm_mag'].iloc[0]:.3f})")
# for ax in axes[-1]:
#     ax.set_xlabel("Date")
# for ax in axes[:, 0]:
#     ax.set_ylabel("y")
# fig.autofmt_xdate()
# fig.suptitle(f"{NUM_PLOT_PER_MAG} examples per anomaly magnitude (gray: base series, dashed: anomaly start)")
# fig.tight_layout()
# plt.show()

# %% Run GaussianPointDetector online
#######################################################################################
############################### Run GaussianPointDetector #############################
#######################################################################################

WARMUP_RATIO = 0.3
CONFIDENCES = [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]  # threshold_param
WEEKS_PER_YEAR = 52

detector = GaussianPointDetector()
initial_len = int(WARMUP_RATIO * num_steps)
anm_starts = anm_info["anm_start"].to_numpy()

# Online detection: y_t is compared with y_{1:t-1}, starting after the warm-up.
# alarms[k, i, t] = 1 if an alarm is triggered at step t of series i with CONFIDENCES[k]
alarms = np.array(
    [
        [detector.detect(pd.Series(ts, index=dates), initial_len, conf).to_numpy() for ts in ts_anm]
        for conf in CONFIDENCES
    ]
)

# False alarms per year: alarms in [initial_len, anm_start), summed over all series
pre_anm = (steps[None, :] >= initial_len) & (labels == 0)
fa_per_year = (alarms * pre_anm).sum(axis=(1, 2)) / (pre_anm.sum() / WEEKS_PER_YEAR)

# Weeks to detection: first alarm at or after anm_start (NaN if never detected)
post_anm_alarms = alarms * labels
detected = post_anm_alarms.any(axis=2)
delay = np.where(detected, post_anm_alarms.argmax(axis=2) - anm_starts, np.nan)

delay_df = pd.DataFrame(
    {
        "confidence": np.repeat(CONFIDENCES, len(anm_info)),
        "mag_norm": np.tile(anm_info["anm_mag_norm"], len(CONFIDENCES)),
        "delay": delay.ravel(),
    }
)
results = (
    delay_df.groupby(["confidence", "mag_norm"])["delay"]
    .agg(
        delay_mean="mean",
        delay_std=lambda d: d.std(ddof=0),
        detection_rate=lambda d: d.notna().mean(),
    )
    .reset_index()
)
results["fa_per_year"] = results["confidence"].map(dict(zip(CONFIDENCES, fa_per_year)))

print(f"Warm-up: {initial_len} steps, false-alarm period: {pre_anm.sum() / WEEKS_PER_YEAR:.1f} years")
print(pd.Series(fa_per_year, index=pd.Index(CONFIDENCES, name="confidence"), name="fa_per_year").round(3))
print(results.round(3).to_string(index=False))

# %% Simulate an improved detector and compare
#######################################################################################
############################### Compare detectors #####################################
#######################################################################################

MAX_YEARS = 3  # z axis cap; 3-D axes do not clip, so values are clipped here
VOLUME_FA_MAX = 7  # same integration domain for every detector, also the plot x range
trapezoid = getattr(np, "trapezoid", None) or np.trapz

# Improved detector (dummy): GaussianPointDetector results with scaled metrics
FA_SCALE, DELAY_MEAN_SCALE, DELAY_STD_SCALE = 0.9, 0.7, 0.5


def surface_grids(results):
    """Mean, mean - std and mean + std of years to detection on the (fa, magnitude) grid."""
    mean_grid = results.pivot(index="mag_norm", columns="fa_per_year", values="delay_mean")
    std_grid = results.pivot(index="mag_norm", columns="fa_per_year", values="delay_std")
    X, Y = np.meshgrid(mean_grid.columns, mean_grid.index)
    Z, Z_std = mean_grid.to_numpy() / WEEKS_PER_YEAR, std_grid.to_numpy() / WEEKS_PER_YEAR
    return X, Y, [np.clip(z, 0, MAX_YEARS) for z in (Z, Z - Z_std, Z + Z_std)]


def volume_under_surface(fa, mags, z, fa_max=VOLUME_FA_MAX, num_points=500):
    """Integrate z[mag, threshold] over false alarms per year in [0, fa_max] and magnitude."""
    order = np.argsort(fa)
    x = np.linspace(0, fa_max, num_points)
    z_x = np.array([np.interp(x, fa[order], z_mag[order]) for z_mag in z])
    return trapezoid(trapezoid(z_x, x, axis=1), mags)


# Sample surfaces: repetition j of every magnitude forms one surface of years to detection
# (missed detections and delays above MAX_YEARS count as MAX_YEARS)
years_to_detection = np.where(
    np.isnan(delay), MAX_YEARS, np.minimum(delay / WEEKS_PER_YEAR, MAX_YEARS)
)
mag_idx = np.searchsorted(MAGNITUDES, anm_info["anm_mag_norm"].round(2))
rep_idx = anm_info.groupby("anm_mag_norm").cumcount().to_numpy()
sample_surfaces = np.empty((NUM_TS_PER_MAG, len(MAGNITUDES), len(CONFIDENCES)))
sample_surfaces[rep_idx, mag_idx] = years_to_detection.T

# Improved detector: scale the mean and the spread around the mean of each grid cell
cell_mean = sample_surfaces.mean(axis=0, keepdims=True)
improved_surfaces = np.clip(
    DELAY_MEAN_SCALE * cell_mean + DELAY_STD_SCALE * (sample_surfaces - cell_mean), 0, MAX_YEARS
)
improved_results = results.assign(
    fa_per_year=results["fa_per_year"] * FA_SCALE,
    delay_mean=results["delay_mean"] * DELAY_MEAN_SCALE,
    delay_std=results["delay_std"] * DELAY_STD_SCALE,
)

detectors = {
    "GaussianPoint": (results, fa_per_year, sample_surfaces, "C0"),
    "Improved": (improved_results, fa_per_year * FA_SCALE, improved_surfaces, "C1"),
}

# Volume under the surface: mean and std over the NUM_TS_PER_MAG sample surfaces
volumes = {
    name: np.array([volume_under_surface(fa, MAGNITUDES, s) for s in surfaces])
    for name, (_, fa, surfaces, _) in detectors.items()
}
for name, v in volumes.items():
    print(f"Volume under surface ({name}): {v.mean():.4f} ± {v.std(ddof=0):.4f}")

# Plot: surfaces (left), volumes (right)
fig = plt.figure(figsize=(17, 8))
grid = fig.add_gridspec(1, 2, width_ratios=[2.3, 1])
ax = fig.add_subplot(grid[0], projection="3d")
for name, (res, _, _, color) in detectors.items():
    X, Y, (Z, Z_low, Z_high) = surface_grids(res)
    ax.plot_surface(X, Y, Z, color=color, edgecolor="white", lw=0.3, alpha=0.6, shade=False)
    for Z_band in (Z_low, Z_high):
        ax.plot_surface(X, Y, Z_band, color=color, alpha=0.1, edgecolor="none", shade=False)
ax.set_xlim(0, VOLUME_FA_MAX)
ax.set_zlim(0, MAX_YEARS)
ax.set_xlabel("False alarms per year")
ax.set_ylabel("Anomaly magnitude")
ax.set_zlabel(f"Years to detection")
ax.legend(
    handles=[Patch(color=color, label=name) for name, (*_, color) in detectors.items()],
    loc="upper left",
)

ax_vol = fig.add_subplot(grid[1])
for i, (name, v) in enumerate(volumes.items()):
    mean, std = v.mean(), v.std(ddof=0)
    ax_vol.bar(i, mean, yerr=std, width=0.5, color=detectors[name][-1], capsize=8)
    ax_vol.text(i, mean + std, f"{mean:.3f} ± {std:.3f}", ha="center", va="bottom")
ax_vol.set_xticks(range(len(volumes)), list(volumes))
ax_vol.set_xlabel("Detector")
ax_vol.set_ylabel("Volume under surface")
ax_vol.set_title(f"Volume under surface")
ax_vol.spines[["top", "right"]].set_visible(False)
ax_vol.grid(axis="y", alpha=0.3)
fig.tight_layout()
plt.show()
