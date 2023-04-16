import sys
sys.path.insert(0, 'C:/Users/lukas/OneDrive - Johannes Kepler Universität Linz/Projekte/DLinear/data')
import data__, experimentSetup, datafactory, data_utils
from importlib import reload
reload(data__)
reload(main_)
reload(datafactory)
reload(data_utils)
from datafactory import DataSet
from dataTransform import Transform
from experimentSetup import Model
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,8)



target = "power_consumption_kwh"
#target = "thermal_consumption_kwh"

experiment = {}

dset = DataSet(start_date="2022-01-01", target=target, scale_target=False, scale_variables=False, time_features=False, dynamic_price=False, demand_price=0.5, feedin_price=0.5).pipeline()

dset = dset[["date",target]]


t = Transform(dataset=dset, resample="h", target=target, scale_X=True)
data= t.transform()
data


from itertools import combinations


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

feature_combinations


from sklearn.model_selection import ParameterGrid
import pandas as pd

params_grid = {
  "epochs": [11],
  "n_hidden1": [500],
  "n_hidden2": [50],
  "features": feature_combinations,
  "batch_size": [64],
  "encoding": ["onehot"],
  "activation1": ["relu"],
  "activation2": ["sigmoid"],
  "lr": [0.001]
}


grid = ParameterGrid(params_grid) 




print(len(grid))
res = []
for p in grid:
    print(p, target)
    model = Model(model="nn", dataset=data[p["features"]], encoding=p["encoding"], scale=True, target=target, test_size=0.05, nn_params=p)
    
    metrics = model.results(plot=False)
    p.update(metrics)
    res.append(p)



df = pd.DataFrame(res)
experiment.update({"df_nn": df})
print(df.sort_values(by='mse', ascending=True))