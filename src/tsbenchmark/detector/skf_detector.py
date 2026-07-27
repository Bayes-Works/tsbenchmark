import pandas as pd
import numpy as np
from typing import Optional
from src.tsbenchmark.detector.base_detector import BaseDetector

class SkfDetector(BaseDetector):
    """
    SKF detector for anomaly detection
    """

    def __init__(
        self,
        anom_threshold: Optional[float] = None,
        model: Optional = None,
    ):

        self._model = model
        super().__init__()
        self._anom_threshold = anom_threshold

    def init_detector(self):
        self._detector_name = "SKF"

    def data_process(self, data, **kwargs):
        covariates = kwargs.get("covariates", [[] for _ in range(len(data))]) 
        _data = {
            "y": data.values,
            "x": covariates,
            "time": data.index,
        }
        return _data

    def anomaly_score(self, data, **kwargs):
        _data = self.data_process(data.copy(), **kwargs)
        scores, _ = self._model.filter(data=_data)

        return pd.DataFrame({"value": scores > self._anom_threshold}, index=data.index)
        









    
