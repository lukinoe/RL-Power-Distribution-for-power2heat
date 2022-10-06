import pickle
import os
import numpy as np
from power_model.utils.onehot_encode import onehot_build_dataset # . = relative path
from power_model.utils.onehot_encode import add_time_features

_dir = os.path.dirname(os.path.realpath(__file__))

with open(_dir + '/power_model/saved_models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(_dir + '/power_model/saved_scaler/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

with open(_dir + '/power_model/saved_scaler/scaler_m1sum.pkl', 'rb') as f:
    scaler_m1sum = pickle.load(f)

target = "power_consumption"

def predict_power(solar_data):

    x = add_time_features(solar_data)
    x = onehot_build_dataset(x, target, del_target=False)
    x[:,0] = scaler_m1sum.transform(x[:,0].reshape(-1, 1)).flatten()

    y_pred = model.predict(x)
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_pred = y_pred.clip(min=0)

    return y_pred



