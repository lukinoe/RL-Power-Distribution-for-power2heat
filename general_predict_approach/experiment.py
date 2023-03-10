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


""" CHANGE TO CURRENT WORKING_DIR """
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)




experiment = {}

#target = "power_consumption_kwh"
target = "thermal_consumption_kwh"

test_size = 0.05






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



def benchmarkModel(grid, experiment):

  print("Lenght Grid: ", len(grid))
  res = []
  for p in grid:
      print(p, target)


      if hasattr(p, 'features'):
          data_in = data[p["features"]]
      else:
          data_in = data
        
      
      model = Model(model=p["model"], dataset=data_in, encoding=p["encoding"], scale=True, target=target, test_size=test_size, model_params=p)
      
      metrics = model.results(plot=False)
      p.update(metrics)
      res.append(p)

  df = pd.DataFrame(res)
  model_name = grid[0]["model"]
  experiment.update({"df_" + model_name : df})
  print(df.sort_values(by='mse', ascending=True))

  return experiment




""" 

  Linear Regression 

"""

params_grid = {
  "model": ["linear_regression"],
  "features": feature_combinations,
  "encoding": ["onehot","cyclical", None]
}

grid = ParameterGrid(params_grid) 

experiment = benchmarkModel(grid, experiment)




""" 

  SVR

"""



params_grid = {
  "model": ["svr"],
  # "kernel": ["rbf", "sigmoid", "poly"],
  # "degree": [3],  # only valid for "poly" kernel
  # "C": [1, 0.8, 0.9],   # default = 1
  # "epsilon": [0.1, 0.03], # default = 0.1
  # "features": feature_combinations,
  "kernel": ["rbf"],
  "degree": [3],  # only valid for "poly" kernel
  "C": [0.8, 1],   # default = 1
  "epsilon": [0.3, 0.1], # default = 0.1
  "encoding": ["cyclical", "onehot", None]
}

grid = ParameterGrid(params_grid) 

experiment = benchmarkModel(grid, experiment)



""" 

  Feed Forward Neural Network

"""




params_grid = {
  "model": ["nn"],
  "epochs": [11],
  "n_hidden1": [100, 250, 500],
  "n_hidden2": [50],
  # "features": feature_combinations,
  "batch_size": [64],
  "encoding": ["onehot"],
  "activation1": ["relu"],
  "activation2": ["sigmoid", "relu"],
  "lr": [0.001]
}
grid = ParameterGrid(params_grid) 

experiment = benchmarkModel(grid, experiment)




""" 

  DLinear 

"""


params_grid = {
  "model": ["DLinear"],
  "n_epochs": [11], 
  "learning_rate": [0.001], 
  "batch_size": [64], 
  "num_layers": [1], 
  "lookback_len": [48,76,100,200], 
  "pred_len": [10,24,30],
  "encoding": [None]
}

grid = ParameterGrid(params_grid) 

experiment = benchmarkModel(grid, experiment)



"""

    LSTM


"""



params_grid = {
  "model": ["lstm"],
  "n_epochs": [15], 
  "learning_rate": [0.001], 
  "batch_size": [64], 
  "hidden_size": [50,75], 
  "num_layers": [1], 
  "lookback_len": [76,100], 
  "pred_len": [10,20],
  "encoding": [None]
}

grid = ParameterGrid(params_grid) 

experiment = benchmarkModel(grid, experiment)



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