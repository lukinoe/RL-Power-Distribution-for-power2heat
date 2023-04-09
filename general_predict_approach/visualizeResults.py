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
order = False
resolution = "h"



dset = DataSet(start_date="2022-01-01", target=target, scale_target=False, scale_variables=False, order=order, time_features=False).pipeline()
dset = dset[["date",target]]
print(dset)
t = Transform(dataset=dset, resample=resolution, target=target, scale_X=True)
data = t.transform()

print(data)


features = list(data.columns)
feature_combinations = get_feature_combinations(features, target)



p = {
  "model": "linear_regression",
  "features": ['val_last_day', 'mean_24h', 'hour', 'day_continuous', target],
  "encoding": "onehot"
}

grid = ParameterGrid(p) 

model = Model(model=p["model"], dataset=data[p["features"]], encoding=p["encoding"], scale=True, target=target, test_size=test_size, shuffle=shuffle, model_params=p)
      
metrics = model.results(plot=True)