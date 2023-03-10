import sys
sys.path.insert(0, 'C:/Users/lukas/OneDrive - Johannes Kepler Universität Linz/Projekte/DLinear/data')
import dataTransform, experimentSetup, datafactory, data_utils

from datafactory import DataSet
from dataTransform import Transform
from experimentSetup import Model
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import ParameterGrid
import pandas as pd
plt.rcParams["figure.figsize"] = (20,8)


#target = "power_consumption_kwh"
target = "thermal_consumption_kwh"

experiment = {}
dset = DataSet(start_date="2022-01-01", target=target, scale_target=False, scale_variables=False, time_features=False, dynamic_price=False, demand_price=0.5, feedin_price=0.5).pipeline()
dset = dset[["date",target]]

t = Transform(dataset=dset, resample="h", target=target, scale_X=True)
data= t.transform()


print(data.head())

data = data[["month", "hour", target]]

features = list(data.columns)
del features[-1]

print(features)

feature_combinations = []
for i in range(len(features)):
    oc = combinations(features, i + 1)
    for c in oc:
        l = list(c)
        l.append(target)
        feature_combinations.append(l)


params_grid = {
  "n_epochs": [20], 
  "learning_rate": [0.001], 
  "batch_size": [64], 
  "num_layers": [1], 
  "lookback_len": [48,76,100,200], 
  "pred_len": [10,24,30],
  "encoding": [None]
}

grid = ParameterGrid(params_grid) 



print(len(grid))
res = []
for p in grid:
    print(p)
    model = Model(model="DLinear", dataset=data, encoding=p["encoding"], scale=True, target=target, test_size=0.05, epochs=200, DLinear_params=p)

    metrics = model.results()
    p.update(metrics)
    res.append(p)

df = pd.DataFrame(res)
experiment.update({"df_DLinear": df})


print(df.sort_values(by='mse', ascending=True))