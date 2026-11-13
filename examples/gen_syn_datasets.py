"""Generate synthetic time series with kernel_synth_gen and plot them.

Usage:
    python examples/gen_syn_ts.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.tsbenchmark.generator.generator_canari import canari_gen
from src.tsbenchmark.generator.generator_kernelsynth import kernel_synth_gen


SAMPLE_FREQUENCY = "W"
NUM_POINTS_PER_CYCLE = 52
NUM_SERIES = 20
MIN_YEARS, MAX_YEARS = 8, 12
SEED = 0
START_DATE = "2001-09-09"  # Sunday, aligned with the "W" (W-SUN) frequency
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
VALUES_PATH = os.path.join(OUTPUT_DIR, "syn_values.csv")
DATETIMES_PATH = os.path.join(OUTPUT_DIR, "syn_datetimes.csv")

rng = np.random.default_rng(SEED)
num_obs_list = rng.integers(
    MIN_YEARS * NUM_POINTS_PER_CYCLE, MAX_YEARS * NUM_POINTS_PER_CYCLE,
    size=NUM_SERIES, endpoint=True,
)

# First half from canari_gen, second half from kernel_synth_gen
generators = [canari_gen] * (NUM_SERIES // 2) + [kernel_synth_gen] * (NUM_SERIES - NUM_SERIES // 2)
series = [
    gen(SAMPLE_FREQUENCY, NUM_POINTS_PER_CYCLE, num_obs=int(n), seed=SEED + i)
    for i, (gen, n) in enumerate(zip(generators, num_obs_list))
]

fig, axes = plt.subplots(5, 4, figsize=(16, 10), sharex=True)
for i, (ts, ax) in enumerate(zip(series, axes.ravel())):
    ax.plot(ts, lw=0.8)
    ax.set_title(
        f"ts_{i} ({generators[i].__name__}): "
        f"{len(ts)} pts = {len(ts) / NUM_POINTS_PER_CYCLE:.1f} yr",
        fontsize=8,
    )
fig.tight_layout()

# Save data
values = {}
datetimes = {}
for i, ts in enumerate(series, start=1):
    dates = pd.date_range(start=START_DATE, periods=len(ts), freq=SAMPLE_FREQUENCY)
    values[f"ts{i}"] = pd.Series(ts, dtype=float)
    datetimes[f"ts{i}"] = pd.Series(dates.strftime("%Y-%m-%d"))

os.makedirs(OUTPUT_DIR, exist_ok=True)
pd.DataFrame(values).to_csv(VALUES_PATH, index=False)
pd.DataFrame(datetimes).to_csv(DATETIMES_PATH, index=False)
print(f"Saved {len(series)} series to {os.path.normpath(VALUES_PATH)}")
print(f"Saved datetimes to {os.path.normpath(DATETIMES_PATH)}")

plt.show()
