"""Regime anomalies: a lasting change from the start step to the end of the series.

Each generator takes the stationary series ts, the start as a ratio of the series length
(start_ratio) and a signed magnitude mag in the units of the series; it returns a new array.
"""

import numpy as np

from src.tsbenchmark.generator.generator_anomaly.anomaly_utils import start_index


def trend_anomaly_gen(
    ts, start_ratio: float, mag: float, num_points_per_year: float
) -> np.ndarray:
    """Add a linear trend of slope mag per year, starting from 0 at the start step."""
    y = np.asarray(ts, dtype=float).copy()
    t_start = start_index(start_ratio, len(y))
    y[t_start:] += mag / num_points_per_year * np.arange(len(y) - t_start)
    return y


def level_anomaly_gen(ts, start_ratio: float, mag: float) -> np.ndarray:
    """Shift the level by mag from the start step."""
    y = np.asarray(ts, dtype=float).copy()
    y[start_index(start_ratio, len(y)):] += mag
    return y
