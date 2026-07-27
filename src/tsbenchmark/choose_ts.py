from typing import Optional
import pandas as pd
import numpy as np

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
