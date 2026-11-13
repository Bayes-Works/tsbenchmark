"""Build a dataset: stationary time series with trend anomalies applied.

Usage:
    from src.tsbenchmark.build_dataset import build_dataset
    series, anomalous, anm_start_ratio, anm_mags = build_dataset(
        ts_path, anm_path, anm_mag_list=[0.1, 0.2, 0.5, 1.0],
        anm_start_range=(0.3, 0.6), num_rep=20, num_points_per_year=52,
        plot_dataset=True,
    )
"""

from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_series(ts_path: str) -> Dict[str, np.ndarray]:
    """Load the stationary time series from a wide CSV, as {name: values}.

    The CSV has one column per series (ts1, ts2, ...); shorter series are padded with
    empty cells at the bottom, so the NaNs are dropped to recover each series' length.
    """
    values = pd.read_csv(ts_path)
    return {name: values[name].dropna().to_numpy() for name in values.columns}


def load_anm_start_ratios(
    anm_path: str, anm_start_range: Tuple[float, float], num_rep: int
) -> List[float]:
    """Load the first num_rep anm_start_ratio values within anm_start_range.

    The CSV has one column, anm_start_ratio: the anomaly start as a ratio of the
    series length.
    """
    anm_info = pd.read_csv(anm_path)
    low, high = anm_start_range
    return [ratio for ratio in anm_info["anm_start_ratio"] if low <= ratio <= high][
        :num_rep
    ]


def build_anm_mags(anm_mag_list: Sequence[float]) -> np.ndarray:
    """Signed anomaly magnitudes from a list of absolute magnitudes: the list is
    sorted and mirrored to the negative side. The magnitudes need not be evenly
    spaced. E.g. [0.1, 0.5, 1.0] gives -1.0, -0.5, -0.1, 0.1, 0.5, 1.0.
    """
    mags = np.unique(np.asarray(anm_mag_list, dtype=float))  # sorted, no duplicates
    if mags.size == 0 or np.any(mags <= 0):
        raise ValueError(
            f"anm_mag_list must hold positive (absolute) magnitudes, got {list(anm_mag_list)}"
        )
    return np.concatenate([-mags[::-1], mags])


def apply_trend_anomaly(
    ts, start_ratio: float, mag: float, num_points_per_year: float
) -> np.ndarray:
    """From start_ratio onward, change the zero trend to a slope of mag
    (unit / num_points_per_year data points). Returns a new array."""
    y = np.asarray(ts, dtype=float).copy()
    t_start = int(round(start_ratio * (len(y) - 1)))
    y[t_start:] += mag / num_points_per_year * np.arange(len(y) - t_start)
    return y


def plot_dataset_figure(
    stationary_series: Dict[str, np.ndarray],
    series_with_anomaly: Dict[str, dict],
    anm_start_ratio: List[float],
    anm_mags: np.ndarray,
    num_cols: int = 4,
) -> None:
    """Plot one subplot per stationary series (black) with ALL its anomalies on top
    (every rep at every magnitude), colored by absolute magnitude so the negative
    side mirrors the positive side's colors. Shows the figure."""
    num_rows = int(np.ceil(len(stationary_series) / num_cols))
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(4 * num_cols, 1.6 * num_rows), sharex=True, squeeze=False
    )
    cmap = plt.get_cmap("viridis")
    abs_min, abs_max = np.abs(anm_mags).min(), np.abs(anm_mags).max()
    abs_span = (abs_max - abs_min) or 1.0  # a single magnitude has zero span
    for name, ax in zip(stationary_series, axes.ravel()):
        for rep in range(len(anm_start_ratio)):
            for mag in anm_mags:
                ax.plot(
                    series_with_anomaly[name][(mag, rep)],
                    lw=0.5,
                    alpha=0.25,
                    color=cmap((abs(mag) - abs_min) / abs_span),
                )
        ax.plot(stationary_series[name], lw=1.8, color="black")
        ax.set_title(
            f"{name}",
            fontsize=8,
        )
    for ax in axes.ravel()[len(stationary_series):]:
        ax.set_visible(False)
    fig.suptitle("Stationary series (black) with all trend anomalies (color = |magnitude|)")
    fig.tight_layout()
    plt.show()


def build_dataset(
    ts_path: str,
    anm_path: str,
    anm_mag_list: Sequence[float],
    anm_start_range: Tuple[float, float],
    num_rep: int,
    num_points_per_year: float,
    plot_dataset: bool = False,
):
    """Load the stationary series and apply every trend anomaly to each of them.

    anm_mag_list holds the absolute anomaly magnitudes to test (any spacing); each is
    applied with both signs. If plot_dataset is True, also plot the dataset (see plot_dataset_figure).

    Returns:
        stationary_series: {name: stationary values}
        series_with_anomaly: {name: {(anm_mag, rep): time series with anomaly}}, one series per
            magnitude and start ratio (rep indexes anm_start_ratio)
        anm_start_ratio: the start ratios used
        anm_mags: the signed magnitudes used (negatives first, then positives)
    """
    stationary_series = load_series(ts_path)
    anm_start_ratio = load_anm_start_ratios(anm_path, anm_start_range, num_rep)
    anm_mags = build_anm_mags(anm_mag_list)

    series_with_anomaly = {
        name: {
            (mag, rep): apply_trend_anomaly(ts, start_ratio, mag, num_points_per_year)
            for mag in anm_mags
            for rep, start_ratio in enumerate(anm_start_ratio)
        }
        for name, ts in stationary_series.items()
    }
    if plot_dataset:
        plot_dataset_figure(stationary_series, series_with_anomaly, anm_start_ratio, anm_mags)

    return stationary_series, series_with_anomaly, anm_start_ratio, anm_mags
