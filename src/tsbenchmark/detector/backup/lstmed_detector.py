import pandas as pd
import numpy as np
from typing import Optional
from src.tsbenchmark.detector.base_detector import BaseDetector
from merlion.models.anomaly.lstm_ed import LSTMEDConfig, LSTMED
from merlion.utils import TimeSeries

class LstmEdDetector(BaseDetector):
    """
    LSTMED detector for anomaly detection
    """

    def __init__(
        self,
        anom_threshold: Optional[float] = None,
        num_epoch: Optional[int] = 50,
        sequence_len: Optional[int] = None,
        hidden_size: Optional[int] = 50,
        batch_size: Optional[int] = None,
        learning_rate: Optional[int] = None,
    ):
        self._num_epoch = num_epoch
        self._sequence_len = sequence_len
        self._hidden_size = hidden_size
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        super().__init__()
        self._anom_threshold = anom_threshold

    def init_detector(self):
        self._detector_name = "LSTMED"
        self._model = LSTMED(LSTMEDConfig(
                        num_epochs=self._num_epoch,
                        sequence_len=self._sequence_len,
                        hidden_size=self._hidden_size,
                        # batch_size=self._batch_size,
                        # lr=self._learning_rate,
        ))

    def data_process(self, data, **kwargs):
        # TODO: data with nan
        # data = data.ffill().bfill()
        # if data.isna().any().any():
        #     data = data.fillna(0)
        return TimeSeries.from_pd(data)
    
    def train(self, data, **kwargs):
        # TODO: data with nan
        # data = data.ffill().bfill().fillna(0)
        train_data = TimeSeries.from_pd(data)
        train_labels = pd.DataFrame(0, index=data.index, columns=["anomaly"])
        train_labels = TimeSeries.from_pd(train_labels)
        self._model.train(train_data=train_data, anomaly_labels=train_labels)

        if self._anom_threshold is None:
            train_scores_ts = self._model.get_anomaly_label(time_series=train_data)
            scores = train_scores_ts.univariates[train_scores_ts.names[0]]
            self._train_scores = scores.to_pd()
            max_train_score = float(scores.max())
            self._anom_threshold = max(1.1 * max_train_score, 0.1)
        
    def anomaly_score(self, data, **kwargs):
        data = self.data_process(data, **kwargs)
        _score = self._model.get_anomaly_label(time_series=data)
        
        scores_uv = _score.univariates[_score.names[0]]
        scores_pd = scores_uv.to_pd()
        
        # Build result DataFrame, True where score exceeds threshold
        score = pd.DataFrame({"value": False}, index=scores_pd.index)
        score.loc[scores_pd.squeeze() > self._anom_threshold, "value"] = True
        
        return score









    
