"""Point anomaly generators: outliers at single time steps.

Following Lai et al. (2021), "Revisiting Time Series Outlier Detection: Definitions and
Benchmarks" (NeurIPS Datasets and Benchmarks):
    - global_outlier_gen: a spike, extreme compared with every other point of the
      series. It lies outside the range [min, max] of the stationary series.
    - contextual_outlier_gen: a glitch, abnormal only for its neighbours. It deviates
      from the value the series would have had at that step but stays inside the
      range [min, max], so a fixed threshold on the whole series cannot find it.

mag is in the units of the series (build_dataset scales it by the series' std first).

Usage:
    from src.tsbenchmark.generator.generator_anomaly.point_anomaly import (
        contextual_outlier_gen,
        global_outlier_gen,
    )
    y = global_outlier_gen(ts, start_ratio=0.4, mag=0.5)
    y, t_anomaly = contextual_outlier_gen(ts, start_ratio=0.4, mag=0.5)
"""

from typing import Tuple

import numpy as np

from src.tsbenchmark.generator.generator_anomaly.anomaly_utils import start_index


def global_outlier_gen(ts, start_ratio: float, mag: float) -> np.ndarray:
    """Global outlier at the start step: |mag| beyond the range of the series.

    For mag > 0 the point becomes max(ts) + mag, for mag < 0 it becomes
    min(ts) + mag, so it is more extreme than every normal point by |mag|.
    Returns a new array.

    Args:
        ts: the stationary time series.
        start_ratio: outlier position, as a ratio of the series length, in [0, 1].
        mag: signed distance beyond the range, in the units of the series.
    """
    if mag == 0:
        raise ValueError("mag must be non-zero for a global outlier")
    y = np.asarray(ts, dtype=float).copy()
    edge = y.max() if mag > 0 else y.min()
    y[start_index(start_ratio, len(y))] = edge + mag
    return y


def contextual_outlier_gen(ts, start_ratio: float, mag: float) -> Tuple[np.ndarray, int]:
    """Contextual outlier: the point deviates by mag from the series' own value, but
    stays inside the range [min, max] of the series.

    The outlier is placed at the start step if x_t + mag stays inside the range;
    otherwise at the nearest step where it does (the earlier one on a tie). So the
    magnitude is exact and the outlier is never a global one, but its position can
    move a few steps from the start step: use the returned index as the label.

    Args:
        ts: the stationary time series.
        start_ratio: requested outlier position, as a ratio of the series length.
        mag: signed deviation from the series' value, in the units of the series.

    Returns:
        y: a new array with the outlier.
        t_anomaly: the time step where the outlier was placed.
    """
    y = np.asarray(ts, dtype=float).copy()
    low, high = y.min(), y.max()
    fits = np.flatnonzero((y + mag >= low) & (y + mag <= high))
    if fits.size == 0:
        raise ValueError(
            f"mag = {mag:.3g} is too large for a contextual outlier: no step where the "
            f"outlier stays inside the series range [{low:.3g}, {high:.3g}]"
        )
    t_requested = start_index(start_ratio, len(y))
    t_anomaly = int(fits[np.argmin(np.abs(fits - t_requested))])
    y[t_anomaly] += mag
    return y, t_anomaly
