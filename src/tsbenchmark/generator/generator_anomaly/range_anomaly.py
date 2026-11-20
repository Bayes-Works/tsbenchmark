"""Range anomalies: an abnormal stretch of ANM_NUM_YEARS years from the start step, cut at
the end of the series.

Each generator takes the stationary series ts, the start as a ratio of the series length
(start_ratio), a magnitude mag in the units of the series and the number of data points in
one year; it returns a new array.
"""

import numpy as np

from tsbenchmark.generator.generator_anomaly.anomaly_utils import start_index

ANM_NUM_YEARS = 3


def range_window(start_ratio: float, num_points_per_year: float, num_obs: int) -> tuple:
    """(start, end) steps of the range, end exclusive."""
    t_start = start_index(start_ratio, num_obs)
    return t_start, min(t_start + round(ANM_NUM_YEARS * num_points_per_year), num_obs)


def noise_anomaly_gen(
    ts, start_ratio: float, mag: float, num_points_per_year: float, seed: int = 0
) -> np.ndarray:
    """Add Gaussian noise of std mag (> 0) over the range.

    The noise is mag * z, with z ~ N(0, 1) drawn from seed: the same seed gives the same
    pattern for every magnitude, only scaled.
    """
    if mag <= 0:
        raise ValueError(f"mag is the std of the added noise and must be > 0, got {mag}")
    y = np.asarray(ts, dtype=float).copy()
    t_start, t_end = range_window(start_ratio, num_points_per_year, len(y))
    z = np.random.default_rng(seed).standard_normal(t_end - t_start)
    y[t_start:t_end] += mag * z
    return y


def yearly_cycle_shape(ts, num_points_per_year: float) -> np.ndarray:
    """Yearly sine wave of amplitude 1, in phase with the series' own yearly cycle.

    Least-squares fit of ts ~ c + a cos(wt) + b sin(wt), with w = 2 pi / num_points_per_year,
    then returns (a cos(wt) + b sin(wt)) / sqrt(a^2 + b^2).
    """
    y = np.asarray(ts, dtype=float)
    t = np.arange(len(y))
    w = 2 * np.pi / num_points_per_year
    X = np.column_stack([np.ones_like(t, dtype=float), np.cos(w * t), np.sin(w * t)])
    _, a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    amplitude = np.hypot(a, b)
    if amplitude == 0:
        raise ValueError("the series has no yearly cycle to scale (fitted amplitude is 0)")
    return (a * np.cos(w * t) + b * np.sin(w * t)) / amplitude


def scale_anomaly_gen(
    ts, start_ratio: float, mag: float, num_points_per_year: float
) -> np.ndarray:
    """Change the amplitude of the yearly cycle by mag over the range.

    Adds mag * yearly_cycle_shape: mag > 0 enlarges the seasonal swing, mag < 0 shrinks it
    (and inverts it if |mag| exceeds the cycle's amplitude).
    """
    y = np.asarray(ts, dtype=float).copy()
    t_start, t_end = range_window(start_ratio, num_points_per_year, len(y))
    shape = yearly_cycle_shape(y, num_points_per_year)
    y[t_start:t_end] += mag * shape[t_start:t_end]
    return y
