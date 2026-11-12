import pandas as pd
import numpy as np
from typing import Optional
from src.tsbenchmark.detector.base_detector import BaseDetector
from time_rcd import TimeRCDDetector


class TimeRcdDetector(BaseDetector):
    """
    Time-RCD detector for anomaly detection (zero-shot, pretrained)
    """

    def __init__(
        self,
        anom_threshold: Optional[float] = 0.5,
        win_size: Optional[int] = 5000,
        variant: Optional[str] = "uni",
    ):
        self._win_size = win_size
        self._variant = variant
        super().__init__()
        self._anom_threshold = anom_threshold

    def init_detector(self):
        self._detector_name = "Time-RCD"
        self._model = TimeRCDDetector.from_pretrained(variant=self._variant, win_size=self._win_size)

    def data_process(self, data, **kwargs):
        # The data is expected to be already resampled; only fill the gaps
        return data.interpolate().bfill()

    def anomaly_score(self, data, **kwargs):
        data = self.data_process(data.copy(), **kwargs)

        scores = np.asarray(self._model.predict(data.iloc[:, 0].to_numpy())).ravel()

        return pd.DataFrame(
            {"value": scores > self._anom_threshold, "score": scores},
            index=data.index,
        )
