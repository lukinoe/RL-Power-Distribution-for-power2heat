import pandas as pd
import numpy as np
from datafactory import DataSet
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,8)


target = "thermal_consumption_kwh"
dset = DataSet(start_date="2022-01-01", target=target, scale_target=False, scale_variables=False, time_features=False, dynamic_price=False, demand_price=0.5, feedin_price=0.5).pipeline()
df = dset[["i_temp1", "i_temp2", "i_temp3", "i_power", "power_consumption_kwh", "thermal_consumption_kwh", "date", "kwh_eq_state"]]
# df[[ "i_metercons", "i_meterfeed", "i_m1sum", "i_m2sum","i_power" ,"power_consumption", "thermal_consumption"]] *= 1000

print(df)