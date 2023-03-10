import sys
import os
sys.path.insert(0, 'C:/Users/lukas/OneDrive - Johannes Kepler Universität Linz/Projekte/DLinear/data')
from datafactory import DataSet
from dataTransform import Transform
from experimentSetup import Model
import matplotlib.pyplot as plt
from itertools import combinations

from sklearn.model_selection import ParameterGrid
import pandas as pd
plt.rcParams["figure.figsize"] = (20,8)


""" CHANGE TO CURRENT WORKING_DIR"""
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)




experiment = {}

#target = "power_consumption_kwh"
target = "thermal_consumption_kwh"


dset = DataSet(start_date="2022-01-01", target=target, scale_target=False, scale_variables=False, time_features=False, dynamic_price=False, demand_price=0.5, feedin_price=0.5).pipeline()
dset = dset[["date",target]]
t = Transform(dataset=dset, resample="h", target=target, scale_X=True)
data = t.transform()



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


""" 

  Linear Regression 

"""

params_grid = {
  "model": ["LinearRegression"],
  "features": feature_combinations,
  "encoding": ["onehot","cyclical", None]
}

grid = ParameterGrid(params_grid) 

print("Lenght Grid: ", len(grid))
res = []
for p in grid:
    print(p, target)
    model = Model(model="linear_regression", dataset=data[p["features"]], encoding=p["encoding"], scale=True, target=target, test_size=0.05, epochs=200, lstm_params=p)
    
    metrics = model.results(plot=False)
    p.update(metrics)
    res.append(p)

df = pd.DataFrame(res)
experiment.update({"df_LinearRegression": df})
print(df.sort_values(by='mse', ascending=True))


""" 

  SVR

"""

params_grid = {
  "model": ["SVR"],
  # "kernel": ["rbf", "sigmoid", "poly"],
  # "degree": [3],  # only valid for "poly" kernel
  # "C": [1, 0.8, 0.9],   # default = 1
  # "epsilon": [0.1, 0.03], # default = 0.1
  "features": feature_combinations,
  "kernel": ["rbf"],
  "degree": [3],  # only valid for "poly" kernel
  "C": [0.8],   # default = 1
  "epsilon": [0.3], # default = 0.1
  "encoding": ["cyclical", "onehot", None]
}

grid = ParameterGrid(params_grid) 


print("Lenght Grid: ", len(grid))
res = []
for p in grid:
    print(p, target)
    model = Model(model="svr", dataset=data[p["features"]], encoding=p["encoding"], scale=True, target=target, test_size=0.05, epochs=200, svr_params=p)
    
    metrics = model.results(plot=False)
    p.update(metrics)
    res.append(p)

df = pd.DataFrame(res)
experiment.update({"df_SVR": df})
print(df.sort_values(by='mse', ascending=True))



""" 

  Feed Forward Neural Network

"""

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


""" 

  DLinear 

"""


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






"""

    LSTM


"""



params_grid = {
  "n_epochs": [11], 
  "learning_rate": [0.001], 
  "batch_size": [64], 
  "hidden_size": [50,75], 
  "num_layers": [1], 
  "lookback_len": [48,76,100], 
  "pred_len": [10],
  "encoding": [None]
}

grid = ParameterGrid(params_grid) 



print(len(grid))
res = []
for p in grid:
    print(p)
    model = Model(model="lstm", dataset=data, encoding=p["encoding"], scale=True, target=target, test_size=0.05, lstm_params=p)

    metrics = model.results()
    p.update(metrics)
    res.append(p)

df = pd.DataFrame(res)
experiment.update({"df_lstm": df})

print(df.sort_values(by='mse', ascending=True))





""" SAVE EXPERIMENTS """

for i,e in enumerate(experiment.items()):
    label = e[0]
    df = e[1]
    print(label)
    print(df.sort_values(by='mse', ascending=True))

    if target == "power_consumption_kwh":
        df.to_csv("benchmarks/power/" + label +  ".csv")
    if target == "thermal_consumption_kwh":
        df.to_csv("benchmarks/thermal/" + label+  ".csv")