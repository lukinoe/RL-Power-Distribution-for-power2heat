import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
_data_path = os.path.join(script_dir, '..', 'data')
sys.path.insert(0, _data_path)
from datafactory import DataSet
from environments.ENV_policyGradient import Environment
from utils import plot_rewards_loss, plot_states
from sklearn.preprocessing import StandardScaler
from MCPolicyFormer import MCPolicyGrad


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


batch_size = 1
seq_len = 96
input_size= 5
hidden_size = 256
lr = 1e-5
output_size= 2
epochs = 200
num_trajectories = 300 # max days: ~ 430
epsilon = 0.0
lr_schedule = 75 # epochs
sampleIdx = False
dynamic_price = True

resample = None
if seq_len == 24:
  resample = "h"


'''
NUM_TRAJECTORIES: important parameter; case = 100: batch_size = 64 --> only 2 updates per epoch --> 64 + 36
'''

max_storage_tank = 18.52

args = {
    "max_storage_tank": max_storage_tank,
    "optimum_storage": max_storage_tank * 0.8,
    "gamma1": 0,    # financial
    "gamma2": 1,      # distance optimum
    "gamma3": 0.0,      # tank change
    "demand_price": 0.5,
    "feedin_price": 0.1
}



dataset = DataSet(start_date="2022-01-01", target="i_m1sum", scale_target=False, scale_variables=False, time_features=False, resample=resample, dynamic_price=dynamic_price, demand_price=args["demand_price"], feedin_price=args["feedin_price"]).pipeline()
dataset["excess"] = (dataset.i_m1sum - dataset.power_consumption_kwh).clip(lower=0)
dataset = dataset[["date", "excess", "demand_price", "feedin_price", "thermal_consumption_kwh", "kwh_eq_state"]]

print(dataset)


SC = StandardScaler()
scaler = SC.fit(dataset[["excess", "demand_price", "feedin_price", "thermal_consumption_kwh", "kwh_eq_state"]].values)

env = Environment(levels=seq_len, max_storage_tank=args["max_storage_tank"], optimum_storage=args["optimum_storage"], gamma1=args["gamma1"], gamma2=args["gamma2"], gamma3=args["gamma3"])
model = MCPolicyGrad(input_size=input_size, hidden_size=hidden_size, output_size=output_size, learning_rate=lr, batch_size=batch_size, num_epochs=1, seq_len=seq_len, dataset=dataset, env=env ,epsilon=epsilon, lr_schedule=lr_schedule, scaler=scaler, sampleIdx=sampleIdx)
day = 160

rewards_list, loss_list, value_loss_list = [], [], []
for i in range(epochs):

    losses, values_losses, states, actions, rewards = model.train()
    
    loss_list.append(losses.numpy())
    value_loss_list.append(values_losses.numpy())
    rewards_list.append(rewards.numpy())
    
    print("Episode " + str(i))
    print("Reward Mean: ", rewards.mean().item())
    print(states[day], actions[day])


plot_rewards_loss(np.array(rewards_list).mean(axis=1), np.array(loss_list).mean(axis=1), np.array(value_loss_list).mean(axis=1), num_trajectories=num_trajectories)


for i in [50,60,75,85,100,115,125,140,150,175,203,204,205,220,225,245,250,260,270]:
    _states, _actions, _reward_sum = model.predict(i)
    plot_states(_states[:,-1], _actions, args["optimum_storage"], id=i)


torch.save(model.model.state_dict(), script_dir + "/checkpoints/model_" + str(seq_len))


    
