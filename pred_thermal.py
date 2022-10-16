import os
import sys
import pickle
from thermal_model.models.DLinear_Merge_Seq2Seq_Timeenc_M  import Model
from thermal_model.utils.timefeatures import time_features
from power_model.utils.onehot_encode import add_time_features
import torch
import numpy as np 
import pandas as pd

_dir = os.path.dirname(os.path.realpath(__file__))

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


d = dotdict()

d.seq_len = 500
d.pred_len = 100
individual = False
channels = 5


model = Model(d)
model.to(device)
model.load_state_dict(torch.load(_dir + "/thermal_model/checkpoints/best/checkpoint.pth", map_location=torch.device(device)))

with open(_dir + '/thermal_model/dataset/saved_scaler/scaler_dict.pkl', 'rb') as f:
    scaler_set = pickle.load(f)


def predict_thermal(df_historic, df_solar):

    df_solar = add_time_features(df_solar, mode="thermal") #attention! minute is not added to encoding!!


    # *** SCALE ***
    df_solar.solar = scaler_set["i_m1sum"].transform(df_solar.solar.to_numpy().reshape(-1,1)).flatten()
    for col in df_historic.columns:
        df_historic[col] = scaler_set[col].transform(df_historic[col].to_numpy().reshape(-1,1)).flatten()


    # *** PREDICT ***
    x_mark = torch.tensor(df_solar.values, dtype=torch.float64).reshape((1,100,5))
    x_historic = torch.tensor(df_historic.values, dtype=torch.float64).reshape((1,500,5))
    #x_mark = torch.cat((torch.randn(1, 100, 4), x_mark), dim=2)

    y_hat = model(x_historic.float(), x_mark.float())
    y_hat = y_hat.detach().numpy()

    y_hat = y_hat.reshape(100,3)

    # *** UNSCALE ***
    for i,col in enumerate(["i_temp1", "i_temp2", "i_temp3"]):
        y_hat[:,i] = scaler_set[col].inverse_transform(y_hat[:,i].reshape(-1,1)).flatten()



    print(y_hat)

    return y_hat
