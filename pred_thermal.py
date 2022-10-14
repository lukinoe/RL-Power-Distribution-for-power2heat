import os
import sys
import inspect
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


def predict_thermal(df_historic, df_solar):

    print(df_historic)

    df_solar = add_time_features(df_solar, mode="thermal") #attention! minute is not added to encoding!!

    # print(df_solar)
    # print(df_solar.values.shape)
    # print(df_historic.values.shape)

    x_mark = torch.tensor(df_solar.values, dtype=torch.float64).reshape((1,100,5))
    x_historic = torch.tensor(df_historic.values, dtype=torch.float64).reshape((1,500,5))
    #x_mark = torch.cat((torch.randn(1, 100, 4), x_mark), dim=2)

    y_hat = model(x_historic.float(), x_mark.float())
    y_hat = y_hat.detach().numpy()

    print(y_hat)

    return y_hat
