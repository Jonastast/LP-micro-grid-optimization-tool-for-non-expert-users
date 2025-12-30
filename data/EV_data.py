import pandas as pd
import numpy as np

df_day = pd.read_csv("data/EV_profile_hourly.csv", parse_dates=["Timestamp"]).set_index("Timestamp")

year_index = pd.date_range("2025-01-01 00:00", periods=24 * 365, freq="H")
repeated = np.tile(df_day.iloc[:24, 0].values, 365)  # repeat the daily profile

df_year = pd.DataFrame({"Timestamp": year_index, df_day.columns[0]: repeated})
df_year.to_csv("data/EV_profile_yearly.csv", index=False)