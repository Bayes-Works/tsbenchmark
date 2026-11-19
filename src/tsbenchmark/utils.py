import numpy as np
import pandas as pd

NUM_POINTS_PER_YEAR = {"daily": 365, "weekly": 52, "monthly": 12, "quarterly": 4}


def get_num_points_per_year(data_sampling_freq):
    """Number of data points in one year for a data sampling frequency."""
    if data_sampling_freq not in NUM_POINTS_PER_YEAR:
        raise ValueError(
            f"Unknown data_sampling_freq {data_sampling_freq!r}; "
            f"choose one of {tuple(NUM_POINTS_PER_YEAR)}"
        )
    return NUM_POINTS_PER_YEAR[data_sampling_freq]


def apply_recursive(data, fn, **kwargs):
    """
    Apply a function recursively through the dictionary data
    """

    if isinstance(data, (pd.Series, pd.DataFrame)):
        return fn(data, **kwargs)
    return {
        key: apply_recursive(value, fn, **kwargs)
        for key, value in data.items()
    }