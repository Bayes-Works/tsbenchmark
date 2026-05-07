from typing import Optional
import pandas as pd
import numpy as np

import pandas as pd

def DataframeToDict (
        data: pd.DataFrame, 
        time: pd.DataFrame
    ) -> dict:
    """
    Create a data dictionary. Each its element is a single time series.
    For example, if the dataset has 100 time series. Will create 
    data_dict[0], data_dict[1], ... , data_dict[99]
    """

    data.columns = data.columns.astype(str)
    time.columns = time.columns.astype(str)
    col = [str(i) for i in range(data.shape[1])]
    data_dict = {}

    for c in col:
        # Get time and values for this column, drop NaN in time
        valid_mask = time[c].notna()
        t = time[c][valid_mask].values
        v = data[c][valid_mask].values

        # Create df_temp with time as index
        df_temp = pd.DataFrame({"value": v}, index=pd.to_datetime(t))
        data_dict[int(c)] = df_temp

    return data_dict

def GenerateAnomaly(
    data: dict,
    anomaly_info,
    col: Optional[list[int]] = None,
):
    """
    Generate synthetic anomalies and add them on top of the provided data.
    anomaly_info contains information about synthetic anomalies.
    col specifies which will be added anomalies.
    """
        
    data_with_anomaly = {}
    num_anomaly_per_magnitude = anomaly_info["num_anomaly_per_magnitude"]
    if col is None:
        col = range(len(data))

    for c in col:
        df_temp = data[c]
        data_with_anomaly[int(c)] = {}
        data_with_anomaly[int(c)]["level"] = {}
        data_with_anomaly[int(c)]["trend"] = {}
        # np.random.seed(anomaly_info["random_seed"])
        # anom_index = np.random.randint(0, len(df_temp.index), size=num_anomaly_per_magnitude)
        anom_index = np.array(anomaly_info["random_number"][:num_anomaly_per_magnitude])
        anom_index = (anom_index * len(df_temp.index)).astype(int).tolist()
        for _, val in enumerate(anomaly_info["anomaly_magnitude"]["level"]):
            data_with_anomaly[int(c)]["level"][str(val)] = {}
            for j in range(num_anomaly_per_magnitude):
                _anomaly_time = df_temp.index[anom_index[j]]
                df_temp_i = df_temp.copy()
                df_temp_i.loc[df_temp_i.index >= _anomaly_time, "value"] += val
                data_with_anomaly[int(c)]["level"][str(val)][j] = df_temp_i

        for _, val in enumerate(anomaly_info["anomaly_magnitude"]["trend"]):
            data_with_anomaly[int(c)]["trend"][str(val)] = {}
            for j in range(num_anomaly_per_magnitude):
                _anomaly_time = df_temp.index[anom_index[j]]
                df_temp_i = df_temp.copy()
                # Apply trend: linearly increasing from anomaly_time onward
                trend_mask = df_temp_i.index >= _anomaly_time
                n = trend_mask.sum()
                df_temp_i.loc[trend_mask, "value"] += val * np.arange(1, n + 1)
                data_with_anomaly[int(c)]["trend"][str(val)][j] = df_temp_i

    return data_with_anomaly