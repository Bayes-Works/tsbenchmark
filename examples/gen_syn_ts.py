"""Generate synthetic time series with kernel_synth_gen and plot them.

Usage:
    python examples/gen_syn_ts.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from generator_canari import canari_gen
from generator_kernelsynth import kernel_synth_gen

SAMPLE_FREQUENCY = "W"
NUM_POINTS_PER_CYCLE = 52
NUM_SERIES = 20
MIN_YEARS, MAX_YEARS = 8, 12
SEED = 0
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "synthetic", "syn_ts.json"
)

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

# Save all series to a JSON file: {"ts_0": [...], "ts_1": [...], ...}
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump({f"ts_{i}": ts for i, ts in enumerate(series)}, f)
print(f"Saved {len(series)} series to {os.path.normpath(OUTPUT_PATH)}")

plt.show()
