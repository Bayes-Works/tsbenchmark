from statistics import NormalDist

import numpy as np
import pandas as pd

from src.tsbenchmark.detector.unified_detector import UnifiedDetector


class GaussianPointDetector(UnifiedDetector):
    """
    Fit a Gaussian to y_{1:t-1}, alarm if y_t is outside its confidence interval.

    - threshold_param: confidence level of the interval, in [0, 1)
    - params:
        ddof: delta degrees of freedom of the standard deviation
    """

    name = "GaussianPoint"
    default_params = {"ddof": 1}

    def _detect(self, y, timestamps, initial_len, threshold_param, params):
        if not 0 <= threshold_param < 1:
            raise ValueError(
                f"threshold_param (confidence) must be in [0, 1), got {threshold_param}"
            )
        z = NormalDist().inv_cdf((1 + threshold_param) / 2)

        alarms = np.zeros(len(y), dtype=int)
        for t in range(max(initial_len, params["ddof"] + 1), len(y)):
            mu = np.mean(y[:t])
            sigma = np.std(y[:t], ddof=params["ddof"])
            if abs(y[t] - mu) > z * sigma:
                alarms[t] = 1
        return alarms


if __name__ == "__main__":
    # Usage example
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2010-01-03", periods=520, freq="W")
    y = rng.normal(10, 1, size=520)
    y[300:] += 0.05 * np.arange(1, 221)  # trend anomaly
    data = pd.Series(y, index=timestamps)

    detector = GaussianPointDetector()
    alarms = detector.detect(data, initial_len=52 * 4, threshold_param=0.95, params={"ddof": 1})
    print(f"{detector.name}: {alarms.sum()} alarms, first at {alarms.idxmax().date()}")
