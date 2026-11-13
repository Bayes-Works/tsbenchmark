"""Generate data/anomaly_info/anm_single_location.json.

Columns:
    - anm_start_ratio: anomaly start, as a ratio of the series length, drawn
      uniformly in [0, 1]

Usage:
    python examples/anm_location_gen.py
"""

import argparse
import json
import os

import numpy as np

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "anomaly_info", "anm_single_location.json"
)


def generate_location(num_samples=1000, seed=0):
    rng = np.random.default_rng(seed)
    anm_start_ratio = rng.uniform(0.0, 1.0, size=num_samples)

    return {
        "anm_start_ratio": np.round(anm_start_ratio, 6).tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num_samples", type=int, default=1000, help="number of rows")
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()

    anm_info = generate_location(args.num_samples, args.seed)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(anm_info, f)

    print(f"Saved {args.num_samples} rows to {os.path.normpath(OUTPUT_PATH)}")
