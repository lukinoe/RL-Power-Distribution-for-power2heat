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
plt.rcParams["figure.figsize"] = (20,8)

dataset = DataSet(start_date="2022-01-01", target="i_m1sum", scale_target=False, scale_variables=False, time_features=False, resample=None,dynamic_price=False, demand_price=0.5, feedin_price=0.5).pipeline()

from extensiveSearch import Tree, Experiment, Experiment_Concat

levels = 24
start_date = "2022-07-24 12:00:00"
# start_date = "2022-06-25 12:00:00"


max_storage_tank = 18.52
args = {
    "max_storage_tank": max_storage_tank,
    "optimum_storage": max_storage_tank * 0.8,
    "gamma1": 0,    # financial
    "gamma2": 1,      # distance optimum
    "gamma3": 0.0,      # tank change
    "demand_price": 0.5,
    "feedin_price": 0.01
}

dataset = DataSet(start_date="2022-01-01", target="i_m1sum", scale_target=False, resample="h", scale_variables=False, time_features=False, dynamic_price=False, demand_price=args["demand_price"], feedin_price=args["feedin_price"]).pipeline()
dataset = dataset[["date", "i_m1sum" , "demand_price", "feedin_price", "power_consumption_kwh", "thermal_consumption_kwh",  "kwh_eq_state"]]
print(dataset[dataset.date == start_date].kwh_eq_state)

e = Experiment(levels, n_samples=2, dataset=dataset, args=args, exploit=True, start_date=start_date) 



res = e.results()
time = np.arange(len(res[0][1]))[1:]
thermal_energy_storage = res[0][0]
optimum_storage_capacity = (np.ones(len(res[0][1])) * (dataset.kwh_eq_state.max() * 0.9))[1:]
action = res[0][1][1:]


print(time.shape, thermal_energy_storage.shape, optimum_storage_capacity.shape, action.shape)


# Visualize the data
fig, ax1 = plt.subplots()

ax1.plot(time, thermal_energy_storage, 'orange', linewidth=2,label='Thermal Energy Storage Capacity')
ax1.plot(time, optimum_storage_capacity, 'g', linestyle='--', label='Optimum Storage Capacity')
ax1.set_xticks(time)
ax1.set_xticklabels(time)
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Storage Capacity (kWh)')
ax1.set_ylim(0, 19)

ax2 = ax1.twinx()
ax2.bar(time, action, alpha=0.3, label='Action (Heat Up)')
ax2.set_ylabel('Action (0 or 1)')
ax2.set_ylim(0, 1)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

plt.title('Thermal Energy Storage Capacity and Action')
plt.show()


