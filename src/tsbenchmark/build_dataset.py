"""Build a dataset: stationary time series with one type of anomaly applied.

Usage:
    from src.tsbenchmark.build_dataset import build_dataset
    series, anomalous, anm_start_ratio, anm_mags = build_dataset(
        ts_path, anm_path, anm_type="trend", anm_mag_list=[0.1, 0.2, 0.5, 1.0],
        anm_start_range=(0.3, 0.6), num_rep=20, num_points_per_year=52,
        plot_dataset=True,
    )
"""

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.tsbenchmark.generator.generator_anomaly.point_anomaly import (
    contextual_outlier_gen,
    global_outlier_gen,
)
from src.tsbenchmark.generator.generator_anomaly.range_anomaly import (
    noise_anomaly_gen,
    scale_anomaly_gen,
)
from src.tsbenchmark.generator.generator_anomaly.regime_anomaly import (
    level_anomaly_gen,
    trend_anomaly_gen,
)

# Anomaly types accepted by build_dataset(anm_type=...), and where each is generated
ANM_TYPES = (
    "no_anomaly",  # the stationary series, unchanged
    "trend",  # regime_anomaly.trend_anomaly_gen
    "level",  # regime_anomaly.level_anomaly_gen
    "global_outlier",  # point_anomaly.global_outlier_gen
    "contextual_outlier",  # point_anomaly.contextual_outlier_gen
    "noise_range",  # range_anomaly.noise_anomaly_gen
    "scale_range",  # range_anomaly.scale_anomaly_gen
)


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


def check_anm_type(anm_type: str) -> None:
    """Raise a ValueError if anm_type is not one of ANM_TYPES."""
    if anm_type not in ANM_TYPES:
        raise ValueError(f"Unknown anm_type {anm_type!r}; choose one of {list(ANM_TYPES)}")


def scale_mag(mag: float, ts) -> float:
    """Convert a magnitude from the standardized space of ts to the units of ts.

    In the standardized space ts has std 1, so a magnitude of mag there is
    mag * std(ts) in the original units (population std, as in (ts - mean) / std).
    """
    return mag * float(np.std(ts))


def apply_anomaly(
    anm_type: str, ts, start_ratio: float, scaled_mag: float, num_points_per_year: float
) -> np.ndarray:
    """Apply one anomaly of type anm_type to ts, by calling its generator.

    scaled_mag is already in the units of ts (see scale_mag). Each generator takes
    only the inputs it needs, e.g. the level shift does not depend on
    num_points_per_year.
    """
    check_anm_type(anm_type)
    if anm_type == "no_anomaly":
        return np.asarray(ts, dtype=float).copy()  # start_ratio and mag are ignored
    if anm_type == "trend":
        return trend_anomaly_gen(ts, start_ratio, scaled_mag, num_points_per_year)
    if anm_type == "level":
        return level_anomaly_gen(ts, start_ratio, scaled_mag)
    if anm_type == "global_outlier":
        return global_outlier_gen(ts, start_ratio, scaled_mag)
    if anm_type == "contextual_outlier":
        return contextual_outlier_gen(ts, start_ratio, scaled_mag)
    if anm_type == "noise_range":
        return noise_anomaly_gen(ts, start_ratio, scaled_mag)
    if anm_type == "scale_range":
        return scale_anomaly_gen(ts, start_ratio, scaled_mag)
    raise AssertionError(f"anm_type {anm_type!r} is in ANM_TYPES but has no generator")


def plot_dataset_figure(
    stationary_series: Dict[str, np.ndarray],
    series_with_anomaly: Dict[str, dict],
    anm_start_ratio: List[float],
    anm_mags: np.ndarray,
    anm_type: str = "",
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
    fig.suptitle(
        f"Stationary series (black) with all {anm_type} anomalies (color = |magnitude|)"
    )
    fig.tight_layout()
    plt.show()


def build_dataset(
    ts_path: str,
    anm_path: str,
    *,
    anm_type: str,
    anm_mag_list: Optional[Sequence[float]] = None,
    anm_start_range: Tuple[float, float],
    num_rep: int,
    num_points_per_year: float,
    plot_dataset: bool = False,
):
    """Load the stationary series and apply every anomaly of type anm_type to each of them.

    anm_type is one of ANM_TYPES ("no_anomaly", "trend", "level", "global_outlier",
    "contextual_outlier", "noise_range", "scale_range"); "no_anomaly" leaves every
    series unchanged; it needs no anm_mag_list (None or empty), and gives one series
    per start ratio, keyed with magnitude 0.0. For every other type anm_mag_list holds the absolute anomaly magnitudes to test (any spacing); each is
    applied with both signs. If plot_dataset is True, also plot the dataset (see plot_dataset_figure).

    The magnitudes are relative: they are in the standardized space of each stationary
    series (std = 1). Before being passed to the anomaly generator, each magnitude is
    scaled to the series' own units: scaled_mag = anm_mag * std(series). So the same
    anm_mag is equally large, relative to its series, in every series.

    Returns:
        stationary_series: {name: stationary values}
        series_with_anomaly: {name: {(anm_mag, rep): time series with anomaly}}, one series per
            magnitude and start ratio (rep indexes anm_start_ratio; anm_mag is the
            relative, unscaled magnitude)
        anm_start_ratio: the start ratios used
        anm_mags: the signed relative magnitudes used (negatives first, then positives)
    """
    # Check the inputs before loading anything
    check_anm_type(anm_type)
    if anm_type == "no_anomaly":
        # The magnitude has no effect: a single magnitude of 0 gives one unchanged
        # series per start ratio, instead of identical copies for every magnitude
        anm_mags = np.array([0.0])
    else:
        if anm_mag_list is None or len(anm_mag_list) == 0:
            raise ValueError(f"anm_mag_list is required for anm_type {anm_type!r}")
        anm_mags = build_anm_mags(anm_mag_list)

    stationary_series = load_series(ts_path)
    anm_start_ratio = load_anm_start_ratios(anm_path, anm_start_range, num_rep)

    series_with_anomaly = {}
    for name, ts in stationary_series.items():
        series_with_anomaly[name] = {}
        for mag in anm_mags:
            scaled_mag = scale_mag(mag, ts)  # relative magnitude -> units of this series
            for rep, start_ratio in enumerate(anm_start_ratio):
                series_with_anomaly[name][(mag, rep)] = apply_anomaly(
                    anm_type, ts, start_ratio, scaled_mag, num_points_per_year
                )
    if plot_dataset:
        plot_dataset_figure(
            stationary_series, series_with_anomaly, anm_start_ratio, anm_mags, anm_type
        )

    return stationary_series, series_with_anomaly, anm_start_ratio, anm_mags
