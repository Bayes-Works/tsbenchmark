"""Range anomaly generators: an abnormal stretch of the series over a range of time steps.

    - noise_anomaly_gen: Gaussian noise is added over the range
    - scale_anomaly_gen: the amplitude of the yearly cycle changes over the range

The range starts at the step given by start_ratio and lasts ANM_NUM_YEARS years (cut at
the end of the series). mag is in the units of the series (build_dataset scales it by
the series' std first).

Usage:
    from src.tsbenchmark.generator.generator_anomaly.range_anomaly import noise_anomaly_gen
    y = noise_anomaly_gen(ts, start_ratio=0.4, mag=0.5, num_points_per_year=52)
    y = scale_anomaly_gen(ts, start_ratio=0.4, mag=0.5, num_points_per_year=52)
"""

import numpy as np

from src.tsbenchmark.generator.generator_anomaly.anomaly_utils import start_index

ANM_NUM_YEARS = 3


def range_window(start_ratio: float, num_points_per_year: float, num_obs: int) -> tuple:
    """(start, end) time steps of a range anomaly lasting ANM_NUM_YEARS years, end
    exclusive and cut at num_obs."""
    t_start = start_index(start_ratio, num_obs)
    return t_start, min(t_start + round(ANM_NUM_YEARS * num_points_per_year), num_obs)


def noise_anomaly_gen(
    ts, start_ratio: float, mag: float, num_points_per_year: float, seed: int = 0
) -> np.ndarray:
    """Add Gaussian noise with std mag over ANM_NUM_YEARS years from the start step.
    Returns a new array.

    The noise is mag * z with z ~ N(0, 1) drawn from seed, so with the same seed every
    magnitude gets the same noise pattern, only scaled.

    Args:
        ts: the stationary time series.
        start_ratio: anomaly start, as a ratio of the series length, in [0, 1].
        mag: std of the added noise, in the units of the series (> 0).
        num_points_per_year: number of data points in one year.
        seed: random seed for the noise.
    """
    if mag <= 0:
        raise ValueError(f"mag is the std of the added noise and must be > 0, got {mag}")
    y = np.asarray(ts, dtype=float).copy()
    t_start, t_end = range_window(start_ratio, num_points_per_year, len(y))
    z = np.random.default_rng(seed).standard_normal(t_end - t_start)
    y[t_start:t_end] += mag * z
    return y


def yearly_cycle_shape(ts, num_points_per_year: float) -> np.ndarray:
    """Unit-amplitude yearly cycle in phase with the series' own yearly cycle.

    Fits ts ~ c + a cos(w t) + b sin(w t) by least squares, with w = 2 pi /
    num_points_per_year (the first Fourier harmonic of the yearly cycle), and returns
    (a cos(w t) + b sin(w t)) / sqrt(a^2 + b^2): a sine wave of amplitude 1 whose
    peaks fall on the peaks of the series' yearly cycle.
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
    """Change the amplitude of the yearly cycle by mag over ANM_NUM_YEARS years from the
    start step. Returns a new array.

    A yearly sine wave of amplitude |mag|, in phase with the series' own yearly cycle
    (see yearly_cycle_shape), is added inside the window: mag > 0 makes the seasonal
    swing bigger, mag < 0 makes it smaller (and inverts it if |mag| exceeds the
    cycle's amplitude).

    Args:
        ts: the stationary time series.
        start_ratio: anomaly start, as a ratio of the series length, in [0, 1].
        mag: signed change of the yearly cycle's amplitude, in the units of the series.
        num_points_per_year: number of data points in one year, i.e. the period of
            the yearly cycle.
    """
    y = np.asarray(ts, dtype=float).copy()
    t_start, t_end = range_window(start_ratio, num_points_per_year, len(y))
    shape = yearly_cycle_shape(y, num_points_per_year)
    y[t_start:t_end] += mag * shape[t_start:t_end]
    return y
