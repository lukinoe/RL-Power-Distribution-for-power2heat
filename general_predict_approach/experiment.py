import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
_data_path = os.path.join(script_dir, '..', 'data')
sys.path.insert(0, _data_path)

from datafactory import DataSet


from dataTransform import Transform
from data_utils import get_feature_combinations
from experimentSetup import Model
import matplotlib.pyplot as plt

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

test_size = 0.10
shuffle = False
resolution = "h"



dset = DataSet(start_date="2022-01-01", target=target, scale_target=False, scale_variables=False, time_features=False).pipeline()
dset = dset[["date",target]]
t = Transform(dataset=dset, resample=resolution, target=target, scale_X=True)
data = t.transform()

print(data)


features = list(data.columns)
feature_combinations = get_feature_combinations(features, target)



def benchmarkModel(grid, experiment):

  print("Lenght Grid: ", len(grid))
  res = []
  for p in grid:
      print(p, target)

      model = Model(model=p["model"], dataset=data[p["features"]], encoding=p["encoding"], scale=True, target=target, test_size=test_size, shuffle=shuffle, model_params=p)
      
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
  "features": feature_combinations,
  "kernel": ["rbf"],
  "degree": [3],  # only valid for "poly" kernel
  "C": [1, 0.8],   # default = 1
  "epsilon": [0.3, 0.1,0.03], # default = 0.1
  "encoding": ["onehot", "cyclical", None]
}

grid = ParameterGrid(params_grid) 

#experiment = benchmarkModel(grid, experiment)



""" 

  Feed Forward Neural Network

"""




params_grid = {
  "model": ["nn"],
  "epochs": [20],
  "n_hidden1": [500],
  "n_hidden2": [50],
  "features": feature_combinations,
  "batch_size": [64],
  "encoding": ["onehot", "cyclical", None],
  "activation1": ["relu"],
  "activation2": ["sigmoid"],
  "lr": [0.001]
}
grid = ParameterGrid(params_grid) 

#experiment = benchmarkModel(grid, experiment)




""" 

  DLinear 

"""


params_grid = {
  "model": ["DLinear"],
  "n_epochs": [7,15,25], 
  "features": [[target]],
  "learning_rate": [0.001, 0.0001], 
  "batch_size": [64], 
  "num_layers": [1], 
  "lookback_len": [75,150,250], 
  "pred_len": [24],
  "encoding": [None]
}

grid = ParameterGrid(params_grid) 

#experiment = benchmarkModel(grid, experiment)



"""

    LSTM


"""



params_grid = {
  "model": ["lstm"],
  "n_epochs": [10], 
  "features": [["month", "hour", "weekday", "day_continuous", target]],
  "learning_rate": [0.001], 
  "batch_size": [64], 
  "hidden_size": [150], 
  "num_layers": [1], 
  "lookback_len": [76,250,], 
  "pred_len": [24],
  "encoding": [None]
}

grid = ParameterGrid(params_grid) 

#experiment = benchmarkModel(grid, experiment)



""" SAVE EXPERIMENTS """
# dfMerge = pd.DataFrame()
# cols_ = ["features", "batch_size", "encoding", "mse", "mape", "mae", "r2"]
# dfMerge.columns = ["model", "features", "batch_size", "encoding", "mse", "mape", "mae", "r2"]
for i,e in enumerate(experiment.items()):
    
    label = e[0]
    df = e[1]

    print(label)
    print(df.sort_values(by='mse', ascending=True))

    if target == "power_consumption_kwh":
        df.to_csv("benchmarks/power/" + label +  ".csv", sep=";")
    if target == "thermal_consumption_kwh":
        df.to_csv("benchmarks/thermal/" + label+  ".csv", sep=";")