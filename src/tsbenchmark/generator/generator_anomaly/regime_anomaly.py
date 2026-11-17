"""Regime anomaly generators: a lasting change in the series from a start point onward.

    - trend_anomaly_gen: the zero trend becomes a slope of mag
    - level_anomaly_gen: the level shifts by mag

Usage:
    from src.tsbenchmark.generator.generator_anomaly.regime_anomaly import (
        level_anomaly_gen,
        trend_anomaly_gen,
    )
    y = trend_anomaly_gen(ts, start_ratio=0.4, mag=0.5, num_points_per_year=52)
    y = level_anomaly_gen(ts, start_ratio=0.4, mag=0.5)
"""

import numpy as np

from src.tsbenchmark.generator.generator_anomaly.anomaly_utils import start_index


def trend_anomaly_gen(
    ts, start_ratio: float, mag: float, num_points_per_year: float
) -> np.ndarray:
    """From start_ratio onward, change the zero trend to a slope of mag
    (unit / num_points_per_year data points). Returns a new array.

    Args:
        ts: the stationary time series.
        start_ratio: anomaly start, as a ratio of the series length, in [0, 1].
        mag: signed slope of the new trend, in units per num_points_per_year points.
        num_points_per_year: number of data points in one year.
    """
    y = np.asarray(ts, dtype=float).copy()
    t_start = start_index(start_ratio, len(y))
    y[t_start:] += mag / num_points_per_year * np.arange(len(y) - t_start)
    return y


def level_anomaly_gen(ts, start_ratio: float, mag: float) -> np.ndarray:
    """From start_ratio onward, shift the level of the series by mag (a step change).
    Returns a new array.

    Args:
        ts: the stationary time series.
        start_ratio: anomaly start, as a ratio of the series length, in [0, 1].
        mag: signed size of the level shift, in the units of the series.
    """
    y = np.asarray(ts, dtype=float).copy()
    y[start_index(start_ratio, len(y)):] += mag
    return y
