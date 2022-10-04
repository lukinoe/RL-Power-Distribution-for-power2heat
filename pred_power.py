import pickle
import os
import pandas as pd
from power_model.utils.onehot_encode import onehot_build_dataset # . = relative path
from power_model.utils.onehot_encode import add_time_features

_dir = os.path.dirname(os.path.realpath(__file__))

with open(_dir + '/power_model/saved_models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(_dir + '/power_model/saved_scaler/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


target = "power_consumption"

def predict_power(solar_data):

    x = add_time_features(solar_data)
    x = onehot_build_dataset(x, target, del_target=False)
    print(x.shape)

    y_pred = model.predict(x)
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

    return y_pred

