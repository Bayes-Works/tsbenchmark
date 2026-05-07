import numpy as np
import pandas as pd

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
):
    prob = {}
    time = {}
    num_anomaly_per_magnitude = anomaly_info["num_anomaly_per_magnitude"] = anomaly_info["num_anomaly_per_magnitude"]
    for ts in anomaly_score:
        prob[ts] = {}
        time[ts] = {}

        for anomaly_type in anomaly_score[ts]:
            prob[ts][anomaly_type] = {}
            time[ts][anomaly_type] = {}

            for _, key in enumerate(anomaly_info["anomaly_magnitude"][anomaly_type]):
                count_anomaly = 0
                sum_time_detection = pd.Timedelta(0)
                # np.random.seed(anomaly_info["random_seed"])
                # ts_len = len(anomaly_score[ts][anomaly_type][str(key)][0].index)
                # anom_index = np.random.randint(0, ts_len, size=num_anomaly)
                ts_len = len(anomaly_score[ts][anomaly_type][str(key)][0].index)
                anom_index = np.array(anomaly_info["random_number"][:num_anomaly_per_magnitude])
                anom_index = (anom_index * ts_len).astype(int).tolist()
                
                for j, df_score in anomaly_score[ts][anomaly_type][str(key)].items():
                    _anomaly_time = df_score.index[anom_index[j]]
                    after_anomaly = df_score.loc[df_score.index >= _anomaly_time, df_score.columns[0]]
                    if after_anomaly.any():
                        count_anomaly += 1
                        first_dec_time = after_anomaly.index[after_anomaly.values][0]
                        _time_to_detection = first_dec_time - _anomaly_time
                        sum_time_detection += _time_to_detection

                prob[ts][anomaly_type][str(key)] = count_anomaly/num_anomaly_per_magnitude
                time[ts][anomaly_type][str(key)] = sum_time_detection/num_anomaly_per_magnitude
        
    return prob, time
                