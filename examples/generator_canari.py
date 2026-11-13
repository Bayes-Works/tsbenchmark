"""Canari generator for a single time series.

Same interface as generator_kernelsynth.kernel_synth_gen, but the series is simulated
from a canari state-space model (as in generate_ts_canari.py) with only two kinds of
components: periodic and autoregression.

The periodic components are stacked and chosen like the periodic kernel bank in
generator_kernelsynth: a main cycle of num_points_per_cycle (P) steps with amplitude 5,
alone, or stacked with shorter calendar periods of sample_frequency that fit strictly
inside the cycle (1 < period < P) — one extra (amplitude ~ U(0, 2.5)) or a pair of
extras (each amplitude ~ U(0, 1.25)). One option is drawn uniformly at random.

The autoregression is randomized like the kernelsynth residual with nu = 0.5:
phi ~ U(0.5, 0.92) and std_error (innovation std) ~ U(0.5, 1), so the AR marginal std
sigma / sqrt(1 - phi^2) stays <= 2.5 at sigma = 1 and the cycle stays visible.

Usage:
    from generator_canari import canari_gen
    ts = canari_gen(sample_frequency="W", num_points_per_cycle=52, seed=0)
"""

from itertools import combinations
from typing import List, Optional

import numpy as np
from canari import Model
from canari.component import Autoregression, Periodic

from generator_kernelsynth import calendar_periods

LENGTH = 1024
PHI_MAX = 0.92  # keeps the AR marginal std <= 2.5 at std_error = 1


def build_periodic_options(
    sample_frequency: str, num_points_per_cycle: float, rng: np.random.Generator
) -> List[List[tuple]]:
    """Options for the stacked periodic components, as lists of (period, amplitude).

    Mirrors the periodic kernel bank of generator_kernelsynth.build_kernel_banks:
    the main cycle alone, plus one extra calendar period, and plus each pair of extras.
    For weekly data with P = 52 the extras are 26, 13, 4 and there are 7 options.
    """
    P = num_points_per_cycle
    extras = sorted(
        {p for p in calendar_periods(sample_frequency).values() if 1 < p < P}, reverse=True
    )

    options = [[(P, 5.0)]]
    options += [[(P, 5.0), (p, rng.uniform(0, 2.5))] for p in extras]
    options += [
        [(P, 5.0), (p1, rng.uniform(0, 1.25)), (p2, rng.uniform(0, 1.25))]
        for p1, p2 in combinations(extras, 2)
    ]
    return options


def canari_gen(
    sample_frequency: str,
    num_points_per_cycle: float,
    num_obs: int = LENGTH,
    seed: Optional[int] = None,
) -> List[float]:
    """Generate one canari time series of num_obs points, as a list.

    Args:
        sample_frequency: pandas frequency string for the data: "h" (hourly), "D"
            (daily), "W" (weekly), "M" (monthly) or "Q" (quarterly). It sets which
            shorter calendar periods can appear inside the cycle.
        num_points_per_cycle: number of time steps in the main cycle for that
            frequency. E.g. weekly data: 52 = yearly cycle, 13 = seasonal cycle;
            monthly data: 12 = yearly cycle.
        num_obs: number of points in the series (default LENGTH = 1024).
        seed: random seed; None gives a different series each call.

    Returns:
        The generated time series as a list of floats.
    """
    if num_points_per_cycle < 2:
        raise ValueError(
            f"num_points_per_cycle must be >= 2 steps, got {num_points_per_cycle}"
        )
    if num_obs < 2:
        raise ValueError(f"num_obs must be >= 2, got {num_obs}")

    rng = np.random.default_rng(seed)

    # Randomly pick one stack of periodic components and the AR parameters
    options = build_periodic_options(sample_frequency, num_points_per_cycle, rng)
    periodic_stack = options[rng.integers(len(options))]
    phi = rng.uniform(0.5, PHI_MAX)
    sigma = rng.uniform(0.5, 1.0)

    # Components, as in generate_ts_canari.py
    components = [
        Periodic(
            period=period,
            mu_states=[amplitude, 0.0],
            var_states=[1e-12, 1e-12],
            std_error=0.0,
        )
        for period, amplitude in periodic_stack
    ]
    components.append(
        Autoregression(phi=phi, std_error=sigma, mu_states=[0.0], var_states=[1e-12])
    )
    model = Model(*components)

    # Simulate from the state-space model: x_t = A x_{t-1} + w_t, y_t = C x_t
    A = np.asarray(model.transition_matrix)
    Q = np.asarray(model.process_noise_matrix)
    C = np.asarray(model.observation_matrix).reshape(1, -1)
    x = np.asarray(model.mu_states).reshape(-1, 1)

    y = np.zeros(num_obs)
    for t in range(num_obs):
        w = rng.multivariate_normal(np.zeros(A.shape[0]), Q).reshape(-1, 1)
        x = A @ x + w
        y[t] = (C @ x).item()

    return y.tolist()
