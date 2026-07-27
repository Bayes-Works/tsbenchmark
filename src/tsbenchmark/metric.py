import numpy as np
import pandas as pd
from typing import Optional

def FalseRate(
    anomaly_score,
):
    
    false_rate = {}
    for ts, df_score in anomaly_score.items():
        count_false = int(df_score.values.sum())
        time_span = df_score.index[-1] - df_score.index[0]
        time_span = time_span/pd.Timedelta(days=365.25)
        false_rate[ts] = count_false / time_span

    return false_rate


def ProbTimeDetection(
    anomaly_score,
    anomaly_info,
    max_anom_detect_time: Optional[pd.Timedelta] = None,
):
    prob = {}
    time = {}

    for ts in anomaly_score:
        prob[ts] = {}
        time[ts] = {}

        for anomaly_type in anomaly_score[ts]:
            prob[ts][anomaly_type] = {}
            time[ts][anomaly_type] = {}

            for key in anomaly_score[ts][anomaly_type]:
                count_anomaly = 0
                num_anomaly_per_magnitude = len(anomaly_score[ts][anomaly_type][key])
                sum_time_detection = pd.Timedelta(0)
                
                for j, df_score in anomaly_score[ts][anomaly_type][key].items():
                    _anomaly_time = anomaly_info[ts][anomaly_type][key][j]
                    _anomaly_time_end = _anomaly_time + max_anom_detect_time
                    after_anomaly = df_score.loc[
                        (df_score.index >= _anomaly_time) & (df_score.index <= _anomaly_time_end),
                        df_score.columns[0]
                    ]

                    if after_anomaly.any():
                        count_anomaly += 1
                        first_dec_time = after_anomaly.index[after_anomaly.values][0]
                        _time_to_detection = first_dec_time - _anomaly_time
                        sum_time_detection += _time_to_detection

                prob[ts][anomaly_type][key] = count_anomaly/num_anomaly_per_magnitude
                time[ts][anomaly_type][key] = sum_time_detection/num_anomaly_per_magnitude
        
    return prob, time
                