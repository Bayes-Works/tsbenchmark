import pandas as pd
import numpy as np
import torch
from typing import Optional
from src.tsbenchmark.detector.base_detector import BaseDetector
from tranad.src.models import TranAD
from tranad.src.constants import args 
from tranad.main import backprop, convert_to_windows


class TranAdDetector(BaseDetector):
    """
    TranAD detector for anomaly detection.
    """

    def __init__(
        self,
        num_epoch: Optional[int] = 5,
        anom_threshold_upper: Optional[float] = None,
        anom_threshold_lower: Optional[float] = None,
    ):
        self._num_epoch = num_epoch
        self._anom_threshold_upper = anom_threshold_upper
        self._anom_threshold_lower = anom_threshold_lower
        self._optimizer = None
        self._scheduler = None
        super().__init__()

    def init_detector(self):
        self._detector_name = "TranAD"
        args.model = 'TranAD'
        self._model = None

    def data_process(self, data: pd.DataFrame, **kwargs):
        arr   = data.to_numpy(dtype=np.float64).reshape(len(data), -1)
        dataO = torch.DoubleTensor(arr)
        dataD = convert_to_windows(dataO, self._model)   # [T, w_size, n_feats]
        return dataO, dataD

    def train(self, data: pd.DataFrame, **kwargs):
        n_feats = data.to_numpy(dtype=np.float64).reshape(len(data), -1).shape[1]
        self._model     = TranAD(n_feats).double()
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self._model.lr, weight_decay=1e-5
        )
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, 5, 0.9
        )

        trainO, trainD = self.data_process(data)

        # ── training via backprop ────────────────────────────────────────────
        for epoch in range(1, self._num_epoch + 1):
            lossT, lr = backprop(
                epoch, self._model, trainD, trainO, self._optimizer, self._scheduler
            )

        # ── threshold calibration on training data ───────────────────────────
        self._model.eval()
        loss_train, _ = backprop(
            0, self._model, trainD, trainO, self._optimizer, self._scheduler,
            training=False
        )
        self._anom_threshold_upper = np.nanmax(loss_train[:, 0]) * 1.1
        self._anom_threshold_lower = np.nanmin(loss_train[:, 0]) * 0.9

    def anomaly_score(self, data: pd.DataFrame, **kwargs):
        self._model.eval()
        dataO, dataD = self.data_process(data, **kwargs)
        loss, _ = backprop(
            0, self._model, dataD, dataO, self._optimizer, self._scheduler,
            training=False
        )
        scores = loss[:, 0]

        result = pd.DataFrame({"value": False}, index=data.index)
        result.loc[
            (scores > self._anom_threshold_upper) | (scores < self._anom_threshold_lower),
            "value"
        ] = True
        return result