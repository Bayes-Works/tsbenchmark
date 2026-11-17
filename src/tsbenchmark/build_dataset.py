"""Build a dataset: stationary time series with one type of anomaly applied."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.tsbenchmark.generator.generator_anomaly.anomaly_utils import start_index
from src.tsbenchmark.generator.generator_anomaly.point_anomaly import (
    contextual_outlier_gen,
    global_outlier_gen,
)
from src.tsbenchmark.generator.generator_anomaly.range_anomaly import (
    noise_anomaly_gen,
    range_window,
    scale_anomaly_gen,
)
from src.tsbenchmark.generator.generator_anomaly.regime_anomaly import (
    level_anomaly_gen,
    trend_anomaly_gen,
)

ANM_TYPES = (
    "no_anomaly",
    "trend",
    "level",
    "global_outlier",
    "contextual_outlier",
    "noise_range",
    "scale_range",
)
NUM_POINTS_PER_YEAR = {"daily": 365, "weekly": 52, "monthly": 12, "quarterly": 4}


def get_num_points_per_year(data_sampling_freq):
    """Number of data points in one year for a data sampling frequency."""
    if data_sampling_freq not in NUM_POINTS_PER_YEAR:
        raise ValueError(
            f"Unknown data_sampling_freq {data_sampling_freq!r}; "
            f"choose one of {tuple(NUM_POINTS_PER_YEAR)}"
        )
    return NUM_POINTS_PER_YEAR[data_sampling_freq]


def load_series(ts_path, num_series=None):
    """First num_series columns (all if None) of a wide CSV, one NaN-padded series each."""
    values = pd.read_csv(ts_path).iloc[:, :num_series]
    return {name: values[name].dropna().to_numpy() for name in values.columns}


def load_anm_start_ratios(anm_path, anm_start_range, num_rep):
    """First num_rep anm_start_ratio values within anm_start_range."""
    low, high = anm_start_range
    ratios = pd.read_csv(anm_path)["anm_start_ratio"]
    return [ratio for ratio in ratios if low <= ratio <= high][:num_rep]


def build_anm_mags(anm_type, anm_mag_list):
    """Signed magnitudes: [0] for no_anomaly, positive only for noise_range (a noise std),
    mirrored to both signs otherwise."""
    if anm_type == "no_anomaly":
        return np.array([0.0])
    mags = np.unique(np.asarray([] if anm_mag_list is None else anm_mag_list, dtype=float))
    if mags.size == 0 or np.any(mags <= 0):
        raise ValueError(f"anm_mag_list must hold positive magnitudes, got {anm_mag_list}")
    return mags if anm_type == "noise_range" else np.concatenate([-mags[::-1], mags])


def apply_anomaly(anm_type, ts, start_ratio, mag, num_points_per_year):
    """Apply one anomaly, mag in the units of ts.

    Returns (series with anomaly, (start, end)): the anomalous steps are start to end - 1.
    """
    n = len(ts)
    t = start_index(start_ratio, n)
    if anm_type == "no_anomaly":
        return np.asarray(ts, dtype=float).copy(), (0, 0)
    if anm_type == "trend":
        return trend_anomaly_gen(ts, start_ratio, mag, num_points_per_year), (t, n)
    if anm_type == "level":
        return level_anomaly_gen(ts, start_ratio, mag), (t, n)
    if anm_type == "global_outlier":
        return global_outlier_gen(ts, start_ratio, mag), (t, t + 1)
    if anm_type == "contextual_outlier":
        y, t = contextual_outlier_gen(ts, start_ratio, mag)  # t: where it was actually placed
        return y, (t, t + 1)
    window = range_window(start_ratio, num_points_per_year, n)
    if anm_type == "noise_range":
        return noise_anomaly_gen(ts, start_ratio, mag, num_points_per_year), window
    return scale_anomaly_gen(ts, start_ratio, mag, num_points_per_year), window


def plot_dataset_figure(
    stationary_series, series_with_anomaly, anm_labels, anm_mags, anm_type, num_cols=4
):
    """One subplot per stationary series (black), all its anomalies on top, colored by |mag|,
    with the steps labelled 1 in any of them marked in red along the bottom."""
    num_rows = -(-len(stationary_series) // num_cols)
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(4 * num_cols, 1.6 * num_rows), sharex=True, squeeze=False
    )
    norm = plt.Normalize(np.abs(anm_mags).min(), np.abs(anm_mags).max())
    for ax, (name, ts) in zip(axes.ravel(), stationary_series.items()):
        for (mag, _), y in series_with_anomaly[name].items():
            ax.plot(y, lw=0.5, alpha=0.25, color=plt.cm.viridis(norm(abs(mag))))
        ax.plot(ts, lw=1.8, color="black")
        t_anm = np.flatnonzero(np.any(list(anm_labels[name].values()), axis=0))
        ax.plot(t_anm, np.full(len(t_anm), 0.02), ".", ms=2, color="red",
                transform=ax.get_xaxis_transform())
        ax.set_title(name, fontsize=8)
    for ax in axes.ravel()[len(stationary_series):]:
        ax.set_visible(False)
    fig.suptitle(
        f"Stationary series (black) with all {anm_type} anomalies (color = |magnitude|, "
        "red = label 1)"
    )
    fig.tight_layout()
    plt.show()


def build_dataset(
    ts_path,
    anm_path,
    *,
    anm_type,
    anm_mag_list=None,
    anm_start_range,
    num_rep,
    data_sampling_freq,
    num_series=None,
    plot_dataset=False,
):
    """Apply one anomaly of type anm_type to each stationary series, for every start ratio
    (rep) and every magnitude.

    anm_mag_list holds absolute magnitudes in std of each series: build_anm_mags gives them
    their signs and each one is multiplied by the series' std before use.

    Returns:
        stationary_series: {name: values}
        series_with_anomaly: {name: {(anm_mag, rep): values}}
        anm_labels: {name: {(anm_mag, rep): labels}}, 1 at the anomalous steps, 0 elsewhere
        anm_start_ratio: start ratios used, indexed by rep
        anm_mags: signed magnitudes used, in std of the series
    """
    if anm_type not in ANM_TYPES:
        raise ValueError(f"Unknown anm_type {anm_type!r}; choose one of {ANM_TYPES}")
    num_points_per_year = get_num_points_per_year(data_sampling_freq)
    anm_mags = build_anm_mags(anm_type, anm_mag_list)
    stationary_series = load_series(ts_path, num_series)
    anm_start_ratio = load_anm_start_ratios(anm_path, anm_start_range, num_rep)

    series_with_anomaly, anm_labels = {}, {}
    for name, ts in stationary_series.items():
        series_with_anomaly[name], anm_labels[name] = {}, {}
        for rep, start_ratio in enumerate(anm_start_ratio):
            for mag in anm_mags:
                y, (start, end) = apply_anomaly(
                    anm_type, ts, start_ratio, mag * ts.std(), num_points_per_year
                )
                labels = np.zeros(len(y), dtype=int)
                labels[start:end] = 1
                series_with_anomaly[name][(mag, rep)] = y
                anm_labels[name][(mag, rep)] = labels
    if plot_dataset:
        plot_dataset_figure(stationary_series, series_with_anomaly, anm_labels, anm_mags, anm_type)
    return stationary_series, series_with_anomaly, anm_labels, anm_start_ratio, anm_mags
