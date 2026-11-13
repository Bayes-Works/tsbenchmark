"""KernelSynth generator for a single time series.

One sample from a Gaussian-process prior whose kernel is the sum of one kernel drawn at random
from each of the periodic and residual banks.

The main cycle is num_points_per_cycle (P) time steps: the user picks the cycle for the
data frequency they want. For example, weekly data with P = 52 has a yearly cycle and
P = 13 a seasonal (quarterly) cycle; monthly data with P = 12 has a yearly cycle.

sample_frequency sets which shorter calendar periods can ride on top of the main cycle
in the periodic bank. Each calendar unit (week, month, quarter, half-year, ...) is
converted to time steps for that frequency, and the ones that fit strictly inside the
cycle (longer than 1 step, shorter than P) become the extra periods. For weekly data
with P = 52 that gives 26 (half-year), 13 (quarter) and 4 (month) — the same bank as
generate_ts_kernelsynth.py; for monthly data with P = 12 it gives 6 and 3.

Usage:
    from examples.generator_kernelsynth import kernel_synth_gen
    ts = kernel_synth_gen(sample_frequency="W", num_points_per_cycle=52, seed=0)
"""

from itertools import combinations
from typing import List, Optional

import numpy as np
from pandas.tseries.frequencies import to_offset
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    Matern,
)

LENGTH = 1024

# Calendar periods in time steps, per data frequency (Chronos conventions)
CALENDAR_PERIODS = {
    "h": {"day": 24, "week": 168, "month": 730, "quarter": 2190, "half_year": 4380, "year": 8760},
    "D": {"week": 7, "month": 30, "quarter": 91, "half_year": 182, "year": 365},
    "W": {"month": 4, "quarter": 13, "half_year": 26, "year": 52},
    "M": {"quarter": 3, "half_year": 6, "year": 12},
    "Q": {"half_year": 2, "year": 4},
}


def matern_kernel(phi: float, sigma: float, nu: float = 0.5, num_obs: int = LENGTH) -> Kernel:
    """Matern kernel whose one-step-ahead innovation std is sigma, as in an AR(1).
    """
    if not 0 < phi < 1:
        raise ValueError(f"phi must be in (0, 1), got {phi}")
    matern = Matern(length_scale=-1 / ((num_obs - 1) * np.log(phi)), nu=nu)
    # Matern has variance 1; r = Var(y_t | all past values) / Var(y_t), from the last
    # diagonal entry of the Cholesky factor (r = 1 - phi^2 for nu = 0.5)
    r = np.linalg.cholesky(matern(np.linspace(0, 1, num_obs)[:, None]))[-1, -1] ** 2
    return ConstantKernel(sigma**2 / r) * matern


def periodic_kernel(
    period: float, amplitude: float, length_scale: float = 1.0, num_obs: int = LENGTH
) -> Kernel:
    """Periodic kernel with a period of `period` steps and std `amplitude`.
    """
    return ConstantKernel(amplitude**2) * ExpSineSquared(
        length_scale=length_scale, periodicity=period / (num_obs - 1)
    )


def calendar_periods(sample_frequency: str) -> dict:
    """Calendar periods in time steps for a pandas frequency string (e.g. "W", "D", "M")."""
    base = to_offset(sample_frequency).name.split("-")[0].upper()
    alias = {"H": "h", "B": "D", "ME": "M", "MS": "M", "QE": "Q", "QS": "Q"}.get(base, base)
    if alias not in CALENDAR_PERIODS:
        raise ValueError(
            f"Unsupported sample_frequency {sample_frequency!r}; "
            f"supported: {sorted(CALENDAR_PERIODS)} (hourly, daily, weekly, monthly, quarterly)"
        )
    return CALENDAR_PERIODS[alias]


def build_kernel_banks(
    sample_frequency: str,
    num_points_per_cycle: float,
    rng: np.random.Generator,
    num_obs: int = LENGTH,
):
    """Build the periodic and residual kernel banks.

    A a main cycle of P steps with amplitude 5, alone or combined with shorter calendar periods — 
    the calendar units of sample_frequency that fit strictly inside the cycle (1 < period < P). 
    The bank holds the main cycle alone, plus one extra period (amplitude ~ U(0, 2.5)), and plus each
    pair of extras (each amplitude ~ U(0, 1.25)). For weekly data with P = 52 the extras
    are 26, 13, 4 and the bank is the original 7 entries.
    """
    P = num_points_per_cycle
    extras = sorted(
        {p for p in calendar_periods(sample_frequency).values() if 1 < p < P}, reverse=True
    )

    main = periodic_kernel(period=P, amplitude=5, num_obs=num_obs)
    kernel_bank_periodic = [main]
    kernel_bank_periodic += [
        main + periodic_kernel(period=p, amplitude=rng.uniform(0, 2.5), num_obs=num_obs)
        for p in extras
    ]
    kernel_bank_periodic += [
        main
        + periodic_kernel(period=p1, amplitude=rng.uniform(0, 1.25), num_obs=num_obs)
        + periodic_kernel(period=p2, amplitude=rng.uniform(0, 1.25), num_obs=num_obs)
        for p1, p2 in combinations(extras, 2)
    ]

    # AR-like residual with randomized parameters. nu sets the smoothness
    # (0.5 = AR(1), 1.5 = remembers slope, 2.5 = remembers curvature), phi the
    # memory, sigma ~ U(0.5, 1) the innovation std. The residual's marginal std is
    # sigma / sqrt(r) (see matern_kernel) and grows with phi and nu; the upper
    # bound of phi per nu keeps it <= 2.5 at sigma = 1, i.e. at most half the
    # main cycle's amplitude of 5, so the cycle stays visible. Bounds computed
    # for LENGTH = 1024.
    nu = float(rng.choice([0.5, 1.5, 2.5]))
    phi_max = {0.5: 0.92, 1.5: 0.72, 2.5: 0.64}[nu]
    kernel_bank_residual = [
        matern_kernel(
            phi=rng.uniform(0.5, phi_max),
            sigma=rng.uniform(0.5, 1.0),
            nu=nu,
            num_obs=num_obs,
        ),
    ]

    return kernel_bank_periodic, kernel_bank_residual


def sample_from_gp_prior(kernel: Kernel, X: np.ndarray, random_seed: Optional[int] = None):
    """Draw one sample from a GP prior with covariance `kernel` at input points X."""
    if X.ndim == 1:
        X = X[:, None]
    gpr = GaussianProcessRegressor(kernel=kernel)
    return gpr.sample_y(X, n_samples=1, random_state=random_seed)


def kernel_synth_gen(
    sample_frequency: str,
    num_points_per_cycle: float,
    num_obs: int = LENGTH,
    seed: Optional[int] = None,
) -> List[float]:
    """Generate one KernelSynth time series of num_obs points, as a list.

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
    bank_periodic, bank_residual = build_kernel_banks(
        sample_frequency, num_points_per_cycle, rng, num_obs=num_obs
    )
    X = np.linspace(0, 1, num_obs)
    while True:
        # Randomly pick one kernel from each bank and sum them
        periodic = bank_periodic[rng.integers(len(bank_periodic))]
        residual = bank_residual[rng.integers(len(bank_residual))]
        kernel = periodic + residual

        # Sample a time series from the GP prior
        try:
            ts = sample_from_gp_prior(kernel, X, random_seed=int(rng.integers(2**32)))
        except np.linalg.LinAlgError as err:
            print("Error caught:", err)
            continue
        return ts.squeeze().tolist()
