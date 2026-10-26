"""Generate data/anomaly_info/single_anm_info.json.

Columns:
    - anm_start_ratio: anomaly start, as a ratio of the series length, drawn
      uniformly in [start, end]
    - anm_mag_norm: normalized anomaly magnitude, from 0 to 10 with a step of
      0.01; each magnitude is repeated num_reps times, each repetition with a
      different anm_start_ratio

Usage:
    python examples/anm_info_gen.py --start 0.4 --end 0.9
"""

import argparse
import json
import os

import numpy as np

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "anomaly_info", "single_anm_info.json"
)
MAG_MIN, MAG_MAX, MAG_STEP = 0.0, 10.0, 0.01


def generate_anm_info(start, end, num_reps=100, seed=0):
    if not 0 <= start < end <= 1:
        raise ValueError(f"Need 0 <= start < end <= 1, got start={start}, end={end}")

    num_mags = int(round((MAG_MAX - MAG_MIN) / MAG_STEP)) + 1
    anm_mag_norm = np.round(np.linspace(MAG_MIN, MAG_MAX, num_mags), 2)

    rng = np.random.default_rng(seed)
    anm_start_ratio = rng.uniform(start, end, size=num_mags * num_reps)

    return {
        "anm_start_ratio": np.round(anm_start_ratio, 6).tolist(),
        "anm_mag_norm": np.repeat(anm_mag_norm, num_reps).tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=float, required=True, help="min anm_start_ratio")
    parser.add_argument("--end", type=float, required=True, help="max anm_start_ratio")
    parser.add_argument("--num_reps", type=int, default=100, help="repetitions per magnitude")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    anm_info = generate_anm_info(args.start, args.end, args.num_reps, args.seed)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(anm_info, f)

    print(
        f"Saved {len(anm_info['anm_mag_norm'])} rows "
        f"({len(set(anm_info['anm_mag_norm']))} magnitudes x {args.num_reps} reps) "
        f"to {os.path.normpath(OUTPUT_PATH)}"
    )
