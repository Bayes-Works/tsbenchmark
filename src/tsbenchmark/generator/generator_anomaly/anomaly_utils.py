"""Helpers shared by the anomaly generators."""


def start_index(start_ratio: float, num_obs: int) -> int:
    """Time step where the anomaly starts, from its ratio of the series length."""
    return int(round(start_ratio * (num_obs - 1)))
