import pickle
import os
import pandas as pd
from .utils.onehot_encode import onehot_build_dataset # . = relative path
from .utils.onehot_encode import add_time_features

_dir = os.path.dirname(os.path.realpath(__file__))


with open(_dir + '/saved_models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(_dir + '/saved_scaler/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# data = pd.read_csv(_dir + "/data/wout_m2sum_target_unscaled_date_m1_timefeatures_02-10.csv")[:1000]


target = "power_consumption"

def predict_power(solar_data):

    x = add_time_features(solar_data)
    print(x)
    x = onehot_build_dataset(x, target, del_target=False)

    print(x, x.shape)

    y_pred = model.predict(x)
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

    print("Pred. Shape: ",y_pred.shape)

    return y_pred

#predict()
