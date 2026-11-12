import pandas as pd
import numpy as np
import torch
from typing import Optional
from src.tsbenchmark.detector.base_detector import BaseDetector
from chronos import Chronos2Pipeline


class Chronos2Detector(BaseDetector):
    """
    Chronos-2 detector for anomaly detection.

    Expanding-window forecast: the first `context_frac` of the data is the initial context, each step
    forecasts the next `horizon`, then the context grows by that window (capped at the latest `max_context`).
    Observations outside the [lower, upper] quantile band are anomalies.
    """

    def __init__(
        self,
        model_name: Optional[str] = "amazon/chronos-2",
        quantile_levels: Optional[list] = (0.15, 0.5, 0.85),
        context_frac: Optional[float] = 0.15,
        max_context: Optional[pd.DateOffset] = pd.DateOffset(years=4),
        horizon: Optional[pd.DateOffset] = pd.DateOffset(years=1),
        device: Optional[str] = None,
    ):
        self._model_name = model_name
        self._quantile_levels = list(quantile_levels)
        self._context_frac = context_frac
        self._max_context = max_context
        self._horizon = horizon
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__()

    def init_detector(self):
        self._detector_name = "Chronos-2"
        self._model = Chronos2Pipeline.from_pretrained(self._model_name, device_map=self._device)

    def data_process(self, data, **kwargs):
        # The data is expected to be already resampled on a regular grid. When the observations
        # are on average more than 10 days apart, the gaps are interpolated; otherwise kept as NaN.
        observed = data.iloc[:, 0].dropna().index
        avg_interval = observed.to_series().diff().mean()
        if avg_interval > pd.Timedelta(days=10):
            data = data.interpolate()
        return data

    def predict(self, timestamps, values, h):
        """(lower, median, upper) quantile forecasts of the next `h` steps."""
        context = pd.DataFrame({"item_id": "s", "timestamp": timestamps, "target": values})
        pred = self._model.predict_df(
            context,
            prediction_length=h,
            quantile_levels=self._quantile_levels,
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
        )
        levels = self._quantile_levels
        return (pred[str(levels[0])].to_numpy()[:h],
                pred[str(levels[len(levels) // 2])].to_numpy()[:h],
                pred[str(levels[-1])].to_numpy()[:h])

    def anomaly_score(self, data, **kwargs):
        data = self.data_process(data.copy(), **kwargs)
        index = data.index
        n = len(index)

        series = data.iloc[:, 0].to_numpy(dtype=np.float32)

        # Initial context is never flagged and has no forecast (NaN quantiles)
        anomaly = np.zeros(n, dtype=bool)
        lower_q = np.full(n, np.nan)
        median_q = np.full(n, np.nan)
        upper_q = np.full(n, np.nan)
        start = int(n * self._context_frac)
        while start < n:
            # Number of time steps covered by the horizon from this point
            end = min(max(int(index.searchsorted(index[start] + self._horizon)), start + 1), n)
            # Keep only the latest `max_context` of data before the forecast start
            context_start = int(index.searchsorted(index[start] - self._max_context))
            lower, median, upper = self.predict(index[context_start:start], series[context_start:start], end - start)
            observed = series[start:end]
            anomaly[start:end] = (observed < lower) | (observed > upper)
            lower_q[start:end] = lower
            median_q[start:end] = median
            upper_q[start:end] = upper
            start = end

        return pd.DataFrame(
            {"value": anomaly, "lower": lower_q, "median": median_q, "upper": upper_q},
            index=index,
        )
