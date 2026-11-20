"""Point anomalies: one outlier at a single time step, after Lai et al. (2021), "Revisiting
Time Series Outlier Detection: Definitions and Benchmarks" (NeurIPS Datasets and Benchmarks):
    - global outlier: extreme for the whole series, outside its range [min, max]
    - contextual outlier: abnormal only for its neighbours, inside [min, max], so a fixed
      threshold on the whole series cannot find it

Each generator takes the stationary series ts, the position as a ratio of the series length
(start_ratio) and a signed magnitude mag in the units of the series.
"""

from typing import Tuple

import numpy as np

from tsbenchmark.generator.generator_anomaly.anomaly_utils import start_index


def global_outlier_gen(ts, start_ratio: float, mag: float) -> np.ndarray:
    """Set the start step |mag| beyond the series range: max + mag if mag > 0, min + mag if
    mag < 0. Returns a new array."""
    if mag == 0:
        raise ValueError("mag must be non-zero for a global outlier")
    y = np.asarray(ts, dtype=float).copy()
    edge = y.max() if mag > 0 else y.min()
    y[start_index(start_ratio, len(y))] = edge + mag
    return y


def contextual_outlier_gen(ts, start_ratio: float, mag: float) -> Tuple[np.ndarray, int]:
    """Add mag to one step, keeping it inside the series range [min, max].

    The outlier goes to the start step if x_t + mag fits in the range, otherwise to the
    nearest step where it does (the earlier one on a tie), so it can move a few steps.

    Returns (new array, step where the outlier was placed): use that step as the label.
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
