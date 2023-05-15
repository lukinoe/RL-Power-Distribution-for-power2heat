import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,5.8)
import sys
sys.path.insert(0, 'C:/Users/lukas/OneDrive - Johannes Kepler Universität Linz/Projekte/DLinear/data')
import datafactory
from datafactory import DataSet
from utils import plot_states
plt.rcParams["figure.figsize"] = (20,8)

dataset = DataSet(start_date="2022-01-01", target="i_m1sum", scale_target=False, scale_variables=False, time_features=False, resample=None,dynamic_price=False, demand_price=0.5, feedin_price=0.5).pipeline()

from extensiveSearch import Tree, Experiment, Experiment_Concat

levels = 24
start_date = "2022-07-24 12:00:00"
#start_date = "2022-06-25 12:00:00"
dynamic_prices = True


max_storage_tank = 18.52
args = {
    "max_storage_tank": max_storage_tank,
    "optimum_storage": max_storage_tank * 0.8, #0.8,
    "gamma1": 1.5,    # financial
    "gamma2": 1,      # distance optimum
    "demand_price": 0.5,
    "feedin_price": 0.1
}

dataset = DataSet(start_date="2022-01-01", target="i_m1sum", scale_target=False, resample="h", scale_variables=False, time_features=False, dynamic_price=dynamic_prices, demand_price=args["demand_price"], feedin_price=args["feedin_price"]).pipeline()
dataset = dataset[["date", "i_m1sum" , "demand_price", "feedin_price", "power_consumption_kwh", "thermal_consumption_kwh",  "kwh_eq_state"]]
print(dataset[dataset.date == start_date].kwh_eq_state)

e = Experiment(levels, n_samples=2, dataset=dataset, args=args, exploit=True, start_date=start_date) 


res = e.results()
time = np.arange(len(res[0][1]))[1:]
thermal_energy_storage = res[0][0]
action = res[0][1][1:]


print(time.shape, thermal_energy_storage.shape, action.shape)


plot_states(thermal_energy_storage, action, args["optimum_storage"])



