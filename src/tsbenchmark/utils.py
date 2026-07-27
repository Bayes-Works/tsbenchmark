import numpy as np
import pandas as pd

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