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
# MAGNITUDES = np.round(np.arange(1, 11) * 0.01, 2)  # normalized: 0.01, 0.02, ..., 0.10
MAGNITUDES = np.round(np.arange(1, 11) * 0.1, 2)  # normalized: 0.01, 0.02, ..., 0.10
ANM_START_RANGE = [0.4, 0.9]
print(MAGNITUDES)
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
print(anm_info)
# Print the range of the anm_start_ratio
print(f"Range of anm_start_ratio: {anm_info['anm_start_ratio'].min()} to {anm_info['anm_start_ratio'].max()}")

###########################################################
################ Showcase: range selection ################
###########################################################
# Only get the anomaly with start ratio ranging from ANM_START_RANGE
anm_info_range = anm_info[
    (anm_info["anm_start_ratio"] >= ANM_START_RANGE[0]) &
    (anm_info["anm_start_ratio"] <= ANM_START_RANGE[1])
].reset_index(drop=True)
print(anm_info_range)
print(f"Range of anm_start_ratio after dropout: {anm_info_range['anm_start_ratio'].min()} to {anm_info_range['anm_start_ratio'].max()}")

# ###############################################################################
# ################## Show all the magnitudes ####################################
# ###############################################################################
# mag_all = anm_info["anm_mag_norm"].round(2).unique()
# print(f"All the magnitudes: {mag_all}")

# ###############################################################################
# ## Get all the anm_start_ratio with the magnitude of 0.1 ######################
# ###############################################################################
# mag=0.1
# anm_info_mag = anm_info[anm_info["anm_mag_norm"].round(2) == mag]
# print(anm_info_mag)

# ###############################################################################
# # Get the first NUM_TS_PER_MAG time series of each magnitude ##################
# ###############################################################################
# # Only take the first NUM_TS_PER_MAG of each magnitude
# anm_info_first = (
#     anm_info
#     .groupby("anm_mag_norm", sort=True)
#     .head(1)
#     .reset_index(drop=True)
# )
# print(anm_info_first)

# ###############################################################################
# ############ Example: usage in practice #######################################
# ###############################################################################
# anm_info = (
#     anm_info[(np.isin(anm_info["anm_mag_norm"].round(2), MAGNITUDES)) & 
#              (anm_info["anm_start_ratio"] >= ANM_START_RANGE[0]) & 
#              (anm_info["anm_start_ratio"] <= ANM_START_RANGE[1])
#     ]
#     .groupby("anm_mag_norm", sort=True)
#     .head(NUM_TS_PER_MAG)
#     .reset_index(drop=True)
# )
# print(anm_info)
# print(f"Range of anm_start_ratio after dropout: {anm_info['anm_start_ratio'].min()} to {anm_info['anm_start_ratio'].max()}")
# print(f"All the magnitudes: {anm_info['anm_mag_norm'].round(2).unique()}")
# print(f"Number of time series: {len(anm_info)} ({len(MAGNITUDES)} magnitudes x {NUM_TS_PER_MAG} reps)")


# ###############################################################################
# ##################### TODO: Denormalize #######################################
# ###############################################################################
# # Denormalize: a trend of m per step on (y - mean) / std is m * std per step on y
# anm_info["anm_mag"] = anm_info["anm_mag_norm"] * y_base.std()
# anm_info["anm_start"] = (anm_info["anm_start_ratio"] * num_steps).astype(int)

print('===========================================')
1/0

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

# 3-D plot: x = false alarms per year, y = magnitude, z = weeks to detection (mean ± std)
# Grid: rows = magnitudes, columns = thresholds (sorted by false alarms per year)
mean_grid = results.pivot(index="mag_norm", columns="fa_per_year", values="delay_mean")
std_grid = results.pivot(index="mag_norm", columns="fa_per_year", values="delay_std")
X, Y = np.meshgrid(mean_grid.columns, mean_grid.index)
Z, Z_std = mean_grid.to_numpy() / WEEKS_PER_YEAR, std_grid.to_numpy() / WEEKS_PER_YEAR
MAX_YEARS = 3  # z axis cap; 3-D axes do not clip, so values are clipped here
Z_low, Z_high = np.clip(Z - Z_std, 0, MAX_YEARS), np.clip(Z + Z_std, 0, MAX_YEARS)
Z = np.clip(Z, 0, MAX_YEARS)

# Volume under the surface. Repetition j of every magnitude forms one sample surface
# of years to detection (missed detections and delays above MAX_YEARS count as MAX_YEARS).
# Each surface is integrated over x in [0, VOLUME_FA_MAX] false alarms per year and over
# the magnitudes; the mean and std are taken over the NUM_TS_PER_MAG sample surfaces.
VOLUME_FA_MAX = 7  # same integration domain for every detector, also the plot x range
trapezoid = getattr(np, "trapezoid", None) or np.trapz


def volume_under_surface(fa, mags, z, fa_max=VOLUME_FA_MAX, num_points=500):
    """Integrate z[mag, threshold] over false alarms per year in [0, fa_max] and magnitude."""
    order = np.argsort(fa)
    x = np.linspace(0, fa_max, num_points)
    z_x = np.array([np.interp(x, fa[order], z_mag[order]) for z_mag in z])
    return trapezoid(trapezoid(z_x, x, axis=1), mags)


years_to_detection = np.where(
    np.isnan(delay), MAX_YEARS, np.minimum(delay / WEEKS_PER_YEAR, MAX_YEARS)
)
mag_idx = np.searchsorted(MAGNITUDES, anm_info["anm_mag_norm"].round(2))
rep_idx = anm_info.groupby("anm_mag_norm").cumcount().to_numpy()
sample_surfaces = np.empty((NUM_TS_PER_MAG, len(MAGNITUDES), len(CONFIDENCES)))
sample_surfaces[rep_idx, mag_idx] = years_to_detection.T

volumes = np.array([volume_under_surface(fa_per_year, MAGNITUDES, s) for s in sample_surfaces])
volume_mean, volume_std = volumes.mean(), volumes.std(ddof=0)
print(f"Volume under surface ({detector.name}): {volume_mean:.4f} ± {volume_std:.4f}")

fig = plt.figure(figsize=(17, 8))
grid = fig.add_gridspec(1, 2, width_ratios=[2.3, 1])
ax = fig.add_subplot(grid[0], projection="3d")
ax.plot_surface(X, Y, Z, color="C0", edgecolor="white", lw=0.3, alpha=0.6, shade=False)
for Z_band in (Z_low, Z_high):
    ax.plot_surface(X, Y, Z_band, color="C0", alpha=0.1, edgecolor="none", shade=False)
ax.set_xlim(0, VOLUME_FA_MAX)
ax.set_zlim(0, MAX_YEARS)
ax.set_xlabel("False alarms per year")
ax.set_ylabel("Anomaly magnitude")
ax.set_zlabel(f"Years to detection")
ax.legend(handles=[Patch(color="C0", label=detector.name)], loc="upper left")

# Volume under the surface per detector (lower is better)
ax_vol = fig.add_subplot(grid[1])
ax_vol.bar(0, volume_mean, yerr=volume_std, width=0.5, color="C0", capsize=8)
ax_vol.text(0, volume_mean + volume_std, f"{volume_mean:.3f} ± {volume_std:.3f}", ha="center", va="bottom")
ax_vol.set_xticks([0], [detector.name])
ax_vol.set_xlabel("Detector")
ax_vol.set_ylabel("Volume under surface")
ax_vol.set_title(f"Volume under surface")
ax_vol.set_xlim(-0.75, 0.75)
ax_vol.spines[["top", "right"]].set_visible(False)
ax_vol.grid(axis="y", alpha=0.3)
fig.tight_layout()
plt.show()
