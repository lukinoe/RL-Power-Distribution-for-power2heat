import sys
sys.path.insert(0, 'C:/Users/lukas/OneDrive - Johannes Kepler Universität Linz/Projekte/DLinear/data')
import data__, main_, datafactory
from importlib import reload
reload(data__)
reload(main_)
reload(datafactory)
from datafactory import DataSet
from data__ import Transform
from main_ import Model
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,8)

target = "thermal_consumption_kwh"


dset = DataSet(start_date="2022-01-01", target=target, scale_target=False, scale_variables=False, time_features=False, dynamic_price=False, demand_price=0.5, feedin_price=0.5).pipeline()
#dset = dset[["date","i_m1sum",target]]

dset = dset[["date",target]]
t = Transform(dataset=dset, resample="h", target=target, scale_X=True)
data= t.transform()
print(data)

lstm_params= {"n_epochs": 5, "learning_rate":0.001, "batch_size": 64, "hidden_size": 50, "num_layers": 1, "lookback_len": 100, "pred_len": 24}
model = Model(model="lstm", dataset=data, encoding="onehot", scale=True, target=target, test_size=0.05, epochs=200, lstm_params=lstm_params)

print(model.results())