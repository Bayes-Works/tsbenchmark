"""Generate synthetic time series with KernelSynth.

Adapted from the Chronos KernelSynth script
https://github.com/amazon-science/chronos-forecasting/blob/main/scripts/kernel-synth.py
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

Each series is one sample from a Gaussian-process prior. Its kernel is the sum of one
kernel drawn at random from each of KERNEL_BANK_periodic, KERNEL_BANK_residual and
KERNEL_BANK_stationary_trend.

Changes from the original script:
    - the kernel is one kernel per bank, summed, instead of 1 to max_kernels kernels
      from a single bank combined with randomly chosen + or * operators
    - series are saved to a CSV (date + one column per series) in data/synthetic,
      instead of a GluonTS Arrow file
    - each series has its own seed, so the dataset is reproducible with parallel jobs
    - a few examples are plotted

Usage:
    python examples/generate_ts_kernelsynth.py -N 100 --seed 0
"""

import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    RationalQuadratic,
    WhiteKernel,
    Matern,
)
from tqdm.auto import tqdm

LENGTH = 1024


def matern_kernel(phi: float, sigma: float, nu: float = 0.5) -> Kernel:
    """Matern kernel whose one-step-ahead innovation std is sigma, as in an AR(1).

    The memory is set from phi as for an AR(1): length_scale = -dx / ln(phi), with one step
    dx = 1 / (LENGTH - 1). The innovation is the part of y_t that the past values cannot
    predict; sigma is its std.
        - nu = 0.5: exactly the AR(1) y_t = phi * y_{t-1} + e_t with e_t ~ N(0, sigma^2)
        - nu = 1.5: ARMA(2, 1) when sampled; the series also remembers its slope
        - nu = 2.5: ARMA(3, 2) when sampled; the series also remembers its curvature
    For nu > 0.5 the series is smoother, so the past predicts it better. The same sigma
    then gives a larger overall variance, and phi is below the lag-1 correlation.
    """
    if not 0 < phi < 1:
        raise ValueError(f"phi must be in (0, 1), got {phi}")
    matern = Matern(length_scale=-1 / ((LENGTH - 1) * np.log(phi)), nu=nu)
    # Matern has variance 1; r = Var(y_t | all past values) / Var(y_t), from the last
    # diagonal entry of the Cholesky factor (r = 1 - phi^2 for nu = 0.5)
    r = np.linalg.cholesky(matern(np.linspace(0, 1, LENGTH)[:, None]))[-1, -1] ** 2
    return ConstantKernel(sigma**2 / r) * matern


def trend_kernel(lower: float, upper: float, num_std: float = 3) -> Kernel:
    """Linear-trend kernel whose slope per step is N(0, s^2), with [lower, upper] = ±num_std * s.

    A GP sample has no hard bounds: with num_std = 3, 99.7% of the slopes fall inside
    [lower, upper]. The GP prior has mean 0, so the bounds must be symmetric.
    DotProduct(sigma_0=0) is the line b * x with b ~ N(0, 1) per unit of x. With one step
    dx = 1 / (LENGTH - 1), a slope s per step is (LENGTH - 1) * s per unit of x, so the
    constant ((LENGTH - 1) * s)^2 scales b to a slope std of s per step.
    """
    if not np.isclose(lower, -upper):
        raise ValueError(f"The GP prior has mean 0, so the bounds must be symmetric, got [{lower}, {upper}]")
    slope_std = upper / num_std
    return ConstantKernel(((LENGTH - 1) * slope_std) ** 2) * DotProduct(sigma_0=0.0)


def periodic_kernel(period: float, amplitude: float, length_scale: float = 1.0) -> Kernel:
    """Periodic kernel with a period of `period` steps and std `amplitude`.

    ExpSineSquared has variance 1; the constant scales it to amplitude^2. With one step
    dx = 1 / (LENGTH - 1), a period of `period` steps is period * dx in x. length_scale
    sets the shape within a cycle: small gives a jagged cycle, large a smoother one.
    Part of the variance is a random level shared by the whole series (about 47% at
    length_scale = 1), so the swing within a cycle has a std of about 0.73 * amplitude.
    """
    return ConstantKernel(amplitude**2) * ExpSineSquared(
        length_scale=length_scale, periodicity=period / (LENGTH - 1)
    )


# Periods are in time steps (= weeks for weekly data). The comments give the data
# frequency each period was chosen for in Chronos. Periods longer than 52 steps
# (one year of weekly data) are commented out.

KERNEL_BANK_periodic = [ 
    periodic_kernel(period=52, amplitude=5),
    periodic_kernel(period=52, amplitude=5)+periodic_kernel(period=26, amplitude=np.random.uniform(0, 2.5)),
    periodic_kernel(period=52, amplitude=5)+periodic_kernel(period=13, amplitude=np.random.uniform(0, 2.5)),
    periodic_kernel(period=52, amplitude=5)+periodic_kernel(period=4, amplitude=np.random.uniform(0, 2.5)),
    periodic_kernel(period=52, amplitude=5)+periodic_kernel(period=4, amplitude=np.random.uniform(0, 1.25))+periodic_kernel(period=13, amplitude=np.random.uniform(0, 1.25)),
    periodic_kernel(period=52, amplitude=5)+periodic_kernel(period=4, amplitude=np.random.uniform(0, 1.25))+periodic_kernel(period=26, amplitude=np.random.uniform(0, 1.25)),
    periodic_kernel(period=52, amplitude=5)+periodic_kernel(period=13, amplitude=np.random.uniform(0, 1.25))+periodic_kernel(period=26, amplitude=np.random.uniform(0, 1.25)),
    # periodic_kernel(period=52, amplitude=0),  # Switch off
]

KERNEL_BANK_residual = [ 
    matern_kernel(phi=0.9, sigma=0.5, nu=0.5),  # AR(1), phi = 0.9, innovation std = 0.5
    # matern_kernel(phi=0.9, sigma=0.12, nu=1.5),  # AR(remember slope), phi = 0.9
    # matern_kernel(phi=0.9, sigma=0.034, nu=2.5),  # AR(remember curvature), phi = 0.9
    # matern_kernel(phi=0.9, sigma=0, nu=2.5),  # Switch off
]

upper_trend = 1 / 52  # max slope per step: +1 unit per year of weekly data
lower_trend = -1 / 52  # min slope per step: -1 unit per year of weekly data

KERNEL_BANK_stationary_trend = [
    trend_kernel(lower_trend, upper_trend),  # slope per step ~ N(0, (upper_trend / 3)^2)
    # ConstantKernel(0**2) *DotProduct(sigma_0=0.0), # Switch off
]

START_DATE = "2010-01-03"  # timestamps are arbitrary, as in the original script
FREQ = "W"
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "synthetic", "kernelsynth_ts.csv"
)
NUM_PLOT = 6


def sample_from_gp_prior(kernel: Kernel, X: np.ndarray, random_seed: Optional[int] = None):
    """Draw one sample from a GP prior with covariance `kernel` at input points X."""
    if X.ndim == 1:
        X = X[:, None]
    gpr = GaussianProcessRegressor(kernel=kernel)
    return gpr.sample_y(X, n_samples=1, random_state=random_seed)


def generate_time_series(seed=None) -> np.ndarray:
    """Generate one KernelSynth time series of length LENGTH."""
    rng = np.random.default_rng(seed)
    X = np.linspace(0, 1, LENGTH)
    while True:
        # Randomly pick one kernel from each bank and sum them
        # periodic, residual, trend = (
        #     bank[rng.integers(len(bank))]
        #     for bank in (KERNEL_BANK_periodic, KERNEL_BANK_residual, KERNEL_BANK_stationary_trend)
        # )
        # kernel = periodic + residual + trend

        periodic, residual, trend = (
            bank[rng.integers(len(bank))]
            for bank in (KERNEL_BANK_periodic, KERNEL_BANK_residual, KERNEL_BANK_stationary_trend)
        )
        kernel = periodic + residual + trend

        # Sample a time series from the GP prior
        try:
            ts = sample_from_gp_prior(kernel, X, random_seed=int(rng.integers(2**32)))
        except np.linalg.LinAlgError as err:
            print("Error caught:", err)
            continue
        return ts.squeeze()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate time series with KernelSynth")
    parser.add_argument("-N", "--num-series", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1, help="parallel jobs, -1 = all cores")
    args = parser.parse_args()

    seeds = np.random.SeedSequence(args.seed).spawn(args.num_series)
    series = Parallel(n_jobs=args.n_jobs)(
        delayed(generate_time_series)(seed) for seed in tqdm(seeds)
    )

    # Save: one row per time step, one column per series
    df = pd.DataFrame(np.column_stack(series), columns=[f"ts_{i}" for i in range(args.num_series)])
    df.insert(0, "date", pd.date_range(START_DATE, periods=LENGTH, freq=FREQ))
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {args.num_series} series of length {LENGTH} to {os.path.normpath(OUTPUT_PATH)}")

    # Plot a few examples
    num_plot = min(NUM_PLOT, args.num_series)
    fig, axes = plt.subplots(num_plot, 1, figsize=(12, 5), sharex=True, squeeze=False)
    for i, ax in enumerate(axes[:, 0]):
        ax.plot(df["date"], df[f"ts_{i}"], lw=0.8)
        ax.set_ylabel(f"ts_{i}")
    axes[-1, 0].set_xlabel("Date")
    fig.suptitle("KernelSynth examples (periodic + residual + trend)")
    fig.tight_layout()
    plt.show()
