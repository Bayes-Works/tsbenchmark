import contextlib
import io
import logging
import random
import pandas as pd
import numpy as np
import torch
from typing import Optional, Union
from chronos import BaseChronosPipeline
from tsfm_public import FlowStateForPrediction
from tsfm_public.toolkit import get_model
from tsfm_public.toolkit.time_series_forecasting_pipeline import (
    TimeSeriesForecastingPipeline,
)
from tsfm_public.toolkit.time_series_preprocessor import (
    TimeSeriesPreprocessor,
)
from tsfm_public.toolkit.w1acas import get_forecast_conformal_adaptive_online_score
from src.tsbenchmark.detector.base_detector import BaseDetector
from src.tsbenchmark.detector.chronos2_detector import Chronos2Detector
from transformers.utils import logging as transformers_logging

# The forecasters reload their model on every call: silence the Hugging Face request logs
# ("HTTP Request: GET https://huggingface.co/...") and the "Loading weights" progress bar
logging.getLogger("httpx").setLevel(logging.WARNING)
transformers_logging.disable_progress_bar()


# ----------------------------------------------------------------------------------------------
# Forecasters of the original W1ACAS implementation, copied from
# https://github.com/ibm-granite/granite-tsfm (notebooks/hfdemo/adaptive_conformal_tsad:
# utils.py, forecasters/chronos_forecaster.py, forecasters/granite_tsfm_forecaster.py and
# forecasters/tirex_forecaster.py). Copyright contributors to the TSFM project.
# Only change: `tirex` is imported inside `tirex_forecaster`, so it is needed only for TiRex.
# ----------------------------------------------------------------------------------------------


def create_rolling_forecast_contexts(
    past_values,
    context_length: int,
    prediction_length: int,
    max_context: int | None = None,  # cap for expanding; also validated for fixed
    mode: str = "fixed",  # "fixed" or "expanding"
    step: int = 1,  # stride between contexts
    return_type: str = "torch",  # "torch" or "numpy"
    dtype=torch.float32,
):
    """
    Create contexts and aligned ground-truth horizons.

    Args:
      past_values       : array-like (T,)
      context_length    : context length (fixed length if mode="fixed", minimum if mode="expanding")
      prediction_length : forecast horizon (H)
      max_context       : optional maximum context length (cap). If None, unbounded.
      mode              : "fixed" -> always length == context_length
                          "expanding" -> length grows with t, clipped by max_context,
                                         and emitted only when t >= context_length
      step              : stride between successive end indices
      return_type       : "torch" or "numpy" for outputs
      dtype             : torch dtype for tensors when return_type="torch"

    Returns:
      contexts : list of contexts (len N), each shape (L_i,) where
                 L_i == context_length if mode="fixed", else L_i ∈ [context_length, max_context]
      y_true   : (N, H) GT matrix, NaN where future values unavailable
      end_idx  : (N,) indices of the last obs included in each context
      n_valid  : (N,) number of valid GT steps per row
      ctx_len  : (N,) length of each emitted context (useful when mode="expanding")
    """
    pv = np.asarray(past_values)
    if pv.ndim != 1:
        raise ValueError("past_values must be 1D or 2D")

    T = int(pv.shape[0])
    H = int(prediction_length)
    if T < context_length:
        raise ValueError(f"Need at least context_length={context_length} points, got {T}")

    if max_context is not None:
        if max_context < context_length:
            raise ValueError(f"max_context ({max_context}) must be >= context_length ({context_length}).")
    assert mode in ["fixed", "expanding"], f"Invalid mode '{mode}'. Must be 'fixed' or 'expanding'."

    end_idx = np.arange(context_length - 1, T, step)
    N = len(end_idx)

    # allocate outputs
    if return_type == "torch":
        y_true = torch.full((N, H), float("nan"), dtype=dtype)
        n_valid = torch.zeros(N, dtype=torch.int64)
        ctx_len = torch.zeros(N, dtype=torch.int64)
    else:
        y_true = np.full((N, H), np.nan, dtype=float)
        n_valid = np.zeros(N, dtype=int)
        ctx_len = np.zeros(N, dtype=int)

    contexts = []
    for i, t_end in enumerate(end_idx):
        t = t_end + 1  # exclusive slice end

        if mode == "fixed":
            L = context_length
            if (max_context is not None) and (L > max_context):
                raise ValueError(f"context_length ({context_length}) exceeds max_context ({max_context}).")
            start = t - L
        else:  # expanding
            # grow with t, but clip to max_context if provided
            L = t if max_context is None else min(t, max_context)
            # ensure we never drop below context_length (given t >= context_length, L>=context_length holds)
            L = max(L, context_length)
            start = t - L

        # context slice
        ctx_np = pv[start:t]
        if return_type == "torch":
            ctx = torch.tensor(ctx_np, dtype=dtype)
        else:
            ctx = ctx_np.astype(float, copy=False)
        contexts.append(ctx)

        # ground truth horizon (NaN-padded)
        nv = min(H, T - t)
        if nv > 0:
            if return_type == "torch":
                y_true[i, :nv] = torch.tensor(pv[t : t + nv], dtype=dtype)
            else:
                y_true[i, :nv] = pv[t : t + nv]
        n_valid[i] = nv
        ctx_len[i] = L

    if return_type == "torch":
        end_idx = torch.tensor(end_idx, dtype=torch.int64)

    output = {}
    output["past_values"] = contexts
    output["future_values"] = y_true
    output["end_idx"] = end_idx
    output["n_gt_valid"] = n_valid
    output["context_valid"] = ctx_len

    return output


CHRONOS_MODELS = {
    "chronos-bolt-small": {
        "model_checkpoint": "amazon/chronos-bolt-small",
        "max_context_length": 2048,
    },
    "chronos-2": {
        "model_checkpoint": "amazon/chronos-2",
        "max_context_length": 2048,
    },
}


def chronos_max_context(pipeline):
    """Extract maximum context length from Chronos pipeline config."""
    cfg = getattr(getattr(pipeline, "model", pipeline), "config", None)
    if cfg is None:
        return None

    # Chronos puts it under chronos_config.context_length
    cc = getattr(cfg, "chronos_config", None)
    if isinstance(cc, dict):
        return cc.get("context_length")
    if cc is not None:
        return getattr(cc, "context_length", None)

    # Fallbacks seen in some models
    for k in ("context_length", "max_context_length", "max_position_embeddings"):
        v = getattr(cfg, k, None)
        if v is not None:
            return v
    return None


def chronos_forecaster(
    df,
    timestamp_column,
    target_columns,
    model_name="chronos-bolt-small",
    model_checkpoint=None,
    context_length=50,
    prediction_length=15,
    fixed_context=False,
    # device_map="cpu",
    max_context=None,
    batch_size=128,
):
    """
    Chronos forecaster compatible with granite_tsfm_forecaster interface.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with time series data
    timestamp_column : str
        Name of timestamp column
    target_columns : list of str
        Names of target columns to forecast
    model_name : str, default='chronos-bolt-small'
        Name of Chronos model (used to lookup checkpoint if model_checkpoint is None)
    model_checkpoint : str, optional
        Path or HuggingFace model ID for Chronos model
    context_length : int, default=50
        Context window length
    prediction_length : int, default=15
        Forecast horizon
    fixed_context : bool, default=False
        If True, use fixed context window; if False, use expanding window
    device_map : str, default="cpu"
        Device for inference ("cpu" or "cuda")
    max_context : int, optional
        Maximum context length (auto-detected from model if None)
    batch_size : int, default=128
        Batch size for inference

    Returns
    -------
    dict
        Dictionary with keys:
        - 'y_pred': np.ndarray, shape (n_samples, prediction_length, n_features)
        - 'y_true': np.ndarray, shape (n_samples, prediction_length, n_features)
    """
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    # Get model checkpoint
    if model_checkpoint is None:
        if model_name in CHRONOS_MODELS:
            model_checkpoint = CHRONOS_MODELS[model_name]["model_checkpoint"]
        else:
            raise ValueError(f"Unknown model_name: {model_name}. Provide model_checkpoint explicitly.")

    # Load Chronos pipeline
    pipeline = BaseChronosPipeline.from_pretrained(
        model_checkpoint,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )

    # Determine max context
    if max_context is None:
        max_context = chronos_max_context(pipeline)
        if max_context is None:
            max_context = 2048
    print(f"Using max_context: {max_context}")

    # Extract target values
    past_values = df[target_columns].values
    if len(past_values.shape) == 1:
        past_values = past_values[:, np.newaxis]

    N, F = past_values.shape

    # Process each feature
    y_pred_list = []
    y_true_list = []

    for ix_f in range(F):
        timeseries_univ = past_values[:, ix_f]

        # Create rolling contexts
        output = create_rolling_forecast_contexts(
            timeseries_univ,
            context_length=context_length,
            prediction_length=prediction_length,
            max_context=max_context,
            mode="fixed" if fixed_context else "expanding",
            return_type="torch",
        )

        contexts_ixf = output["past_values"]
        target_values_ixf = output["future_values"]

        # Batch inference
        m_all = []
        num_contexts = len(contexts_ixf)

        for i in range(0, num_contexts, batch_size):
            batch_ctx = contexts_ixf[i : i + batch_size]

            # Get median prediction (quantile 0.5)
            _, batch_m = pipeline.predict_quantiles(
                inputs=batch_ctx,
                prediction_length=prediction_length,
                quantile_levels=[0.5],
            )

            # Convert to numpy
            batch_m = batch_m.detach().cpu().numpy() if hasattr(batch_m, "detach") else np.asarray(batch_m)
            m_all.append(batch_m)

        # Concatenate batch results
        out_m_ixf = np.concatenate(m_all, axis=0) if m_all else np.array([])
        y_pred_list.append(out_m_ixf[..., np.newaxis])

        # Convert target values to numpy
        target_np = (
            target_values_ixf.detach().cpu().numpy()
            if hasattr(target_values_ixf, "detach")
            else np.asarray(target_values_ixf)
        )
        y_true_list.append(target_np[..., np.newaxis])

    # Concatenate features
    y_pred = np.concatenate(y_pred_list, axis=-1)
    y_true = np.concatenate(y_true_list, axis=-1)

    return {"y_pred": y_pred, "y_true": y_true[:, : y_pred.shape[1], :]}


GRANITE_TSFM_CHECKPOINTS = {
    "ttm": "ibm-granite/granite-timeseries-ttm-r2",
    "flowstate": "ibm-granite/granite-timeseries-flowstate-r1",
}


def granite_tsfm_forecaster(
    df,
    timestamp_column,
    target_columns,
    model_name="ttm",
    model_checkpoint=None,
    context_length=50,
    prediction_length=15,
    fixed_context=False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_checkpoint is None:
        model_checkpoint = GRANITE_TSFM_CHECKPOINTS[model_name]
    if model_name == "ttm":
        model = get_model(
            model_checkpoint,
            context_length=context_length,
            prediction_length=prediction_length,
        )
    if model_name == "flowstate":
        model = FlowStateForPrediction.from_pretrained(model_checkpoint)

    ## TimeSeriesPreprocessor
    tsp = TimeSeriesPreprocessor(
        timestamp_column=timestamp_column,
        target_columns=target_columns,
        prediction_length=prediction_length,
        context_length=context_length,
        scaling=False,
    )
    tsp.train(df)
    ## Forecast Pipeline
    fpipe = TimeSeriesForecastingPipeline(model, feature_extractor=tsp, device=device)
    forecast = fpipe(df)
    y_pred = np.array(
        [np.stack(z) for z in forecast[[f"{target_column}_prediction" for target_column in target_columns]].values]
    ).transpose(0, 2, 1)
    y_true = np.array([np.stack(z) for z in forecast[list(target_columns)].values]).transpose(0, 2, 1)

    return {"y_pred": y_pred, "y_true": y_true[:, : y_pred.shape[1], :]}


TIREX_MODELS = {
    "tirex": {
        "model_checkpoint": "NX-AI/TiRex",
        "max_context_length": 2048,
    },
}


def tirex_forecaster(
    df,
    timestamp_column,
    target_columns,
    model_name="tirex",
    model_checkpoint=None,
    context_length=96,
    prediction_length=28,
    fixed_context=False,
    device_map="cpu",
    max_context=None,
    batch_size=128,
):
    """
    TiRex forecaster compatible with granite_tsfm_forecaster and chronos_forecaster interface.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with time series data
    timestamp_column : str
        Name of timestamp column
    target_columns : list of str
        Names of target columns to forecast
    model_name : str, default='tirex'
        Name of TiRex model (used to lookup checkpoint if model_checkpoint is None)
    model_checkpoint : str, optional
        Path or HuggingFace model ID for TiRex model
    context_length : int, default=96
        Context window length
    prediction_length : int, default=28
        Forecast horizon
    fixed_context : bool, default=True
        If True, use fixed context window; if False, use expanding window
    device_map : str, default="cpu"
        Device for inference ("cpu" or "cuda")
    max_context : int, optional
        Maximum context length (uses model default if None)
    batch_size : int, default=128
        Batch size for inference

    Returns
    -------
    dict
        Dictionary with keys:
        - 'y_pred': np.ndarray, shape (n_samples, prediction_length, n_features)
        - 'y_true': np.ndarray, shape (n_samples, prediction_length, n_features)
    """
    from tirex import ForecastModel, load_model  # only needed for TiRex (pip install tirex-ts)

    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Get model checkpoint
    if model_checkpoint is None:
        if model_name in TIREX_MODELS:
            model_checkpoint = TIREX_MODELS[model_name]["model_checkpoint"]
        else:
            raise ValueError(f"Unknown model_name: {model_name}. Provide model_checkpoint explicitly.")

    # Load TiRex model
    pipeline: ForecastModel = load_model(model_checkpoint, device=device_map)

    # Determine max context
    if max_context is None:
        max_context = TIREX_MODELS.get(model_name, {}).get("max_context_length", 2048)
    print(f"Using max_context: {max_context}")

    # Extract target values
    past_values = df[target_columns].values
    if len(past_values.shape) == 1:
        past_values = past_values[:, np.newaxis]

    N, F = past_values.shape

    # Process each feature
    y_pred_list = []
    y_true_list = []

    for ix_f in range(F):
        timeseries_univ = past_values[:, ix_f]

        # Create rolling contexts
        output = create_rolling_forecast_contexts(
            timeseries_univ,
            context_length=context_length,
            prediction_length=prediction_length,
            max_context=max_context,
            mode="fixed" if fixed_context else "expanding",
            return_type="torch",
        )

        contexts_ixf = output["past_values"]
        target_values_ixf = output["future_values"]

        # Batch inference
        m_all = []
        num_contexts = len(contexts_ixf)

        for i in range(0, num_contexts, batch_size):
            batch_ctx = contexts_ixf[i : i + batch_size]

            # Get median prediction (quantile 0.5)
            _, batch_m = pipeline.forecast(
                context=batch_ctx,
                prediction_length=prediction_length,
            )

            # Convert to numpy
            batch_m = batch_m.detach().cpu().numpy() if hasattr(batch_m, "detach") else np.asarray(batch_m)
            m_all.append(batch_m)

        # Concatenate batch results
        out_m_ixf = np.concatenate(m_all, axis=0) if m_all else np.array([])
        y_pred_list.append(out_m_ixf[..., np.newaxis])

        # Convert target values to numpy
        target_np = (
            target_values_ixf.detach().cpu().numpy()
            if hasattr(target_values_ixf, "detach")
            else np.asarray(target_values_ixf)
        )
        y_true_list.append(target_np[..., np.newaxis])

    # Concatenate features
    y_pred = np.concatenate(y_pred_list, axis=-1)
    y_true = np.concatenate(y_true_list, axis=-1)

    return {"y_pred": y_pred, "y_true": y_true[:, : y_pred.shape[1], :]}


# ----------------------------------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------------------------------

MODEL_NAMES = ["chronos-bolt-small", "tirex", "ttm", "flowstate"]


class W1acasDetector(BaseDetector):
    """
    W1ACAS detector for anomaly detection (Wasserstein-1 distance-based Adaptive Conformal
    Anomaly Scoring), using the forecasters of the original implementation (`main_acas_w1.py`).

    Rolling forecast: after the first `context_length` observations, the forecasting model
    (`model_name`: "chronos-bolt-small", "tirex", "ttm" or "flowstate") forecasts the next
    `prediction_length` steps from every time step. W1ACAS turns the forecast errors into online
    conformal p-values, one per forecast step, combined across the steps with `aggregation`.
    Observations with a p-value below `significance_level` are anomalies.
    """

    def __init__(
        self,
        model_name: Optional[str] = "chronos-bolt-small",
        context_length: Optional[int] = 90,
        prediction_length: Optional[int] = 15,
        significance_level: Optional[float] = 0.01,
        aggregation: Optional[str] = "Cauchy",
        nonconformity_score: Optional[str] = "absolute_error",
        n_epochs: Optional[int] = 1,
        n_batch_update: Optional[int] = 10,
        lr: Optional[float] = 0.001,
        prior_past_weights_value: Optional[Union[int, str]] = "proximity",
        seed: Optional[int] = 42,
    ):
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model_name: {model_name}. Must be one of {MODEL_NAMES}.")
        self._model_name = model_name
        self._context_length = context_length
        self._prediction_length = prediction_length
        self._significance_level = significance_level
        self._aggregation = aggregation
        self._nonconformity_score = nonconformity_score
        self._n_epochs = n_epochs
        self._n_batch_update = n_batch_update
        self._lr = lr
        self._prior_past_weights_value = prior_past_weights_value
        self._seed = seed
        super().__init__()
        self._anom_threshold = significance_level

    def init_detector(self):
        self._detector_name = f"W1ACAS ({self._model_name})"
        # Same choice of forecaster as main_acas_w1.py
        if self._model_name in ["ttm", "flowstate"]:
            self._forecaster = granite_tsfm_forecaster
        elif "chronos" in self._model_name:
            self._forecaster = chronos_forecaster
        else:
            self._forecaster = tirex_forecaster

    def data_process(self, data, **kwargs):
        # Same data processing as the Chronos-2 detector
        return Chronos2Detector.data_process(self, data, **kwargs)

    def anomaly_score(self, data, **kwargs):
        data = self.data_process(data.copy(), **kwargs)
        index = data.index
        values = data.iloc[:, 0].to_numpy(dtype=float)
        h = self._prediction_length
        alpha = self._significance_level

        # As in main_acas_w1.py, the missing values are dropped and the time steps are indexed
        # by their position: the models forecast the sequence of observations
        observed = ~np.isnan(values)
        df_input = pd.DataFrame({"index": range(int(observed.sum())), "values": values[observed]})

        n_calibration = int(np.ceil(1 / alpha)) + h
        if len(df_input) - self._context_length < n_calibration:
            raise ValueError(
                f"W1ACAS needs more than {n_calibration} observations after the first "
                f"{self._context_length} (context) for a significance level of {alpha}, "
                f"got {len(df_input) - self._context_length}."
            )

        # Fixed seeds so the weight learning gives the same results on every run
        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)

        # The forecaster and the scoring function print their settings on every call; keep the output quiet
        with contextlib.redirect_stdout(io.StringIO()):
            forecast_output = self._forecaster(
                df_input,
                "index",
                ["values"],
                model_name=self._model_name,
                model_checkpoint=None,
                context_length=self._context_length,
                prediction_length=h,
                fixed_context=False,
            )
            y_pred, y_true = forecast_output["y_pred"], forecast_output["y_true"]
            # The last context forecasts past the end of the data only, so it can't be scored
            if np.all(np.isnan(y_true[-1])):
                y_pred, y_true = y_pred[:-1], y_true[:-1]

            p_values = get_forecast_conformal_adaptive_online_score(
                {"y_pred": y_pred, "y_true": y_true},
                significance_level=alpha,
                aggregation_forecast_horizon=self._aggregation,
                nonconformity_score=self._nonconformity_score,
                forecast_steps=h,
                aggregation_features=None,
                n_epochs=self._n_epochs,
                n_batch_update=self._n_batch_update,
                lr=self._lr,
                prior_past_weights_value=self._prior_past_weights_value,
                align_forecast=True,
            )[:, 0]

        # As in main_acas_w1.py, the p-values are those of the last observations. The initial
        # context and the missing values have no p-value (NaN) and are never flagged.
        score_observed = np.full(len(df_input), np.nan)
        score_observed[-len(p_values):] = p_values
        score = np.full(len(index), np.nan)
        score[observed] = score_observed

        return pd.DataFrame({"value": score < alpha, "score": score}, index=index)
