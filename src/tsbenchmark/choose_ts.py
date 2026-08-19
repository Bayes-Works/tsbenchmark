from typing import Optional
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
import os

### TODO: This file contains funtions to be deleted after finishing choosing time series benchmark

def AddToCsv(df, values_path: str, datetimes_path: str, rewrite: bool = False):
    """
    Checks if a time series's name exists in values.csv and datetimes.csv.
    If not found, appends:
      - df time series's values as a new column to values.csv
      - df time series's index as a new column to datetimes.csv
 
    Args:
        df: DataFrame with 1 column containing the values. Its index include
            datetime 
        values_path: path to values.csv
        datetimes_path: path to datetimes.csv
        rewrite: if True, overwrite the column even if it already exists
    """

    assert len(df.columns) == 1, "df must have only 1 column"
 
    col_name = df.columns[0]
    print(f"Column name to check: '{col_name}'")
 
    # --- Load or initialize values.csv ---
    if os.path.exists(values_path):
        values = pd.read_csv(values_path)
    else:
        print(f"'{values_path}' not found.")
        values = pd.DataFrame()
 
    # --- Load or initialize datetimes.csv ---
    if os.path.exists(datetimes_path):
        datetimes = pd.read_csv(datetimes_path)
    else:
        print(f"'{datetimes_path}' not found.")
        datetimes = pd.DataFrame()
 
    # --- Check if column already exists in both files ---
    in_values = col_name in values.columns
    in_datetimes = col_name in datetimes.columns
 
    if in_values and in_datetimes and not rewrite:
        print(f"Column '{col_name}' already exists in both files. No changes made.")
        return
 
    # --- Add/overwrite values.csv ---
    if not in_values or rewrite:
        if rewrite and in_values:
            values = values.drop(columns=[col_name])
        new_col = pd.DataFrame({col_name: df[col_name].values})
        values = pd.concat([values, new_col], axis=1)
        values.to_csv(values_path, index=False, na_rep="")
        print(f"Added '{col_name}' values to '{values_path}'.")
    else:
        print(f"Column '{col_name}' already in '{values_path}'. Skipped.")
 
    # --- Add/overwrite datetimes.csv ---
    if not in_datetimes or rewrite:
        if rewrite and in_datetimes:
            datetimes = datetimes.drop(columns=[col_name])
        new_col = pd.DataFrame({col_name: df.index.values})
        datetimes = pd.concat([datetimes, new_col], axis=1)
        datetimes.to_csv(datetimes_path, index=False, na_rep="")
        print(f"Added '{col_name}' index to '{datetimes_path}'.")
    else:
        print(f"Column '{col_name}' already in '{datetimes_path}'. Skipped.")

def delete_column(values_path: str, datetimes_path: str, column_name: str):
    for path in (values_path, datetimes_path):
        df = pd.read_csv(path)
        if column_name in df.columns:
            df = df.drop(columns=column_name)
            df.to_csv(path, index=False)
            print(f"Removed '{column_name}' from {path}")
        else:
            print(f"'{column_name}' not found in {path}")

def detrend_prophet(df, standardize=True, **prophet_kwargs):
    """Detrend a single-column time series with Prophet, then standardize.

    Prophet is fit ONLY on observed rows, so no gap-filling guess biases the
    trend. Missing points are then filled from Prophet's own yhat
    (trend + seasonality), which is trend-consistent by construction.

    Input:  df — DataFrame with a date-like index and one value column.
    Output: df_detrend — detrended (and optionally standardized) DataFrame.
                         Gaps filled from yhat when fill_missing=True,
                         left as NaN when False.
            trend    — Prophet trend component (all dates).
            seasonal — Prophet total seasonal component (additive_terms).
    """
    # Reshape into Prophet's expected ds / y format
    df_detrend = df.copy()
    df_detrend = df_detrend.reset_index()
    df_detrend.columns = ["ds", "y"]
    df_detrend["ds"] = pd.to_datetime(df_detrend["ds"])
    df_detrend.index = df_detrend["ds"]

    # Remember where the real NaNs are (no interpolation anywhere)
    missing = df_detrend["y"].isna().values

    # Fit Prophet on observed rows only
    prophet_model = Prophet(**prophet_kwargs)
    prophet_model.fit(df_detrend.loc[~missing, ["ds", "y"]])

    # Predict on ALL dates to get trend / seasonal / yhat everywhere
    prophet_results = prophet_model.predict(df_detrend[["ds"]])
    trend    = prophet_results["trend"]
    seasonal = prophet_results["additive_terms"]

    # Remove the trend; optionally restore NaNs
    detrend_data = df_detrend["y"].values.copy() - trend.values
    df_detrend["y"] = detrend_data

    # Clean up index / helper column
    df_detrend = df_detrend.drop(columns=["ds"], errors="ignore")
    df_detrend.index.name = None

    scale_std = df_detrend.std()
    # Standardize
    if standardize:
        df_detrend = (df_detrend - df_detrend.mean()) / scale_std

    return df_detrend, trend, scale_std

def plot_decomposition(time, original, detrend, trend, figsize=(12, 10)):
    """Plot the detrend decomposition in three stacked panels.

    Inputs:
        original — original data values (array-like or DataFrame/Series)
        time     — x-axis values (the datetime index)
        detrend  — detrended data values
        trend    — the extracted trend component
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    axes[0].plot(time, original)
    axes[0].set_ylabel("original data")

    axes[1].plot(time, trend)
    axes[1].set_ylabel("trend")

    axes[2].plot(time, detrend)
    axes[2].set_ylabel("detrended data")

    fig.tight_layout()
    return fig, axes

def min_max_scale(df):
    """Min-max scale each column to [-1, 1]."""
    lo, hi = df.min(), df.max()
    return 2 * (df - lo) / (hi - lo) - 1

def remove_ma(df, window):
    na_mask = df.isna()
    filled = df.interpolate(method='linear', limit_direction='both')
    # ma = filled.rolling(window).mean()
    # detrended = (filled - ma).mask(na_mask)
    # return detrended
    ma = filled.rolling(window, center=True).mean().fillna(0)
    detrended = (filled - ma).mask(na_mask)
    return detrended
    