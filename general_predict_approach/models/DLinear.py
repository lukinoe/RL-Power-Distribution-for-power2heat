from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
from torch import nn

from re import X
import torch.nn as nn
import torch.nn.functional as F

from sklearn.utils.extmath import cartesian
from sklearn.preprocessing import OneHotEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series

        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    DLinear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Decoder = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                self.Linear_Decoder.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Decoder = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    
        self.time_layer = nn.Linear(self.seq_len,self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        x = x.permute(0,2,1) # to [Batch, Output length, Channel]

        x = x[:,:,-1]

        return x  


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


# class DataDLinear():

#     def __init__(self, X_train, y_train, X_test, y_test, lookback_len=100, pred_len=24) -> None:
        
#         self.lookback_len = lookback_len
#         self.pred_len = pred_len
#         self.generate_sequences(X_train, y_train, X_test, y_test)
        

#     def split_sequences(self,input_sequences, output_sequence):


#         print("***",input_sequences.shape, output_sequence.shape)
#         n_steps_in = self.lookback_len
#         n_steps_out = self.pred_len

#         X, y = list(), list() # instantiate X and y
#         for i in range(len(input_sequences)):
#             # find the end of the input, output sequence
#             end_ix = i + n_steps_in
#             out_end_ix = end_ix + n_steps_out - 1
#             # check if we are beyond the dataset
#             if out_end_ix > len(input_sequences): break
#             # gather input and output of the pattern
#             seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
#             X.append(seq_x), y.append(seq_y)

#         return np.array(X), np.array(y)

#     def generate_sequences(self, X_train, y_train, X_test, y_test):
#         print("***",X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#         self.X_train, self.y_train = self.split_sequences(X_train, y_train)
#         self.X_test, self.y_test = self.split_sequences(X_test, y_test)

        

#     def return_tensors(self):

#         X_train_tensors = Variable(torch.Tensor(self.X_train).to(device))
#         X_test_tensors = Variable(torch.Tensor(self.X_test).to(device))

#         y_train_tensors = Variable(torch.Tensor(self.y_train).to(device))
#         y_test_tensors = Variable(torch.Tensor(self.y_test).to(device))


#         X_train_tensors_final = torch.reshape(X_train_tensors,   
#                                             (X_train_tensors.shape[0], self.lookback_len, 
#                                             X_train_tensors.shape[2]))
#         X_test_tensors_final = torch.reshape(X_test_tensors,  
#                                             (X_test_tensors.shape[0], self.lookback_len, 
#                                             X_test_tensors.shape[2])) 

#         print("Training Shape:", X_train_tensors_final.shape, y_train_tensors.shape)
#         print("Testing Shape:", X_test_tensors_final.shape, y_test_tensors.shape) 

#         return X_train_tensors_final, y_train_tensors,X_test_tensors_final, y_test_tensors,



class DataDLinear():

    def __init__(self, X, y, lookback_len=100, pred_len=24) -> None:
        
        self.lookback_len = lookback_len
        self.pred_len = pred_len
        self.generate_sequences(X,y)
        

    def split_sequences(self,input_sequences, output_sequence):


        print("***",input_sequences.shape, output_sequence.shape)
        n_steps_in = self.lookback_len
        n_steps_out = self.pred_len

        X, y = list(), list() # instantiate X and y
        for i in range(len(input_sequences)):
            # find the end of the input, output sequence
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out - 1
            # check if we are beyond the dataset
            if out_end_ix > len(input_sequences): break
            # gather input and output of the pattern
            seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
            X.append(seq_x), y.append(seq_y)

        return np.array(X), np.array(y)

    def generate_sequences(self, X, y):
        print("***",X.shape, y.shape)
        self.X, self.y = self.split_sequences(X, y)

        

    def return_tensors(self):

        X_tensors = Variable(torch.Tensor(self.X).to(device))
        y_tensors = Variable(torch.Tensor(self.y).to(device))



        X_tensors = torch.reshape(X_tensors,   
                                            (X_tensors.shape[0], self.lookback_len, 
                                            X_tensors.shape[2]))


        return X_tensors, y_tensors





class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



class Trainer_DLinear():

    def __init__(self, params) -> None:

        self.params = params

    def init_network(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        d = dotdict()

        d.seq_len = self.params["lookback_len"]
        d.pred_len = self.params["pred_len"]
        d.individual = False
        channels = self.params["input_size"]

        model = Model(d)
        model.to(device)

        return model
        

    def training_loop(self, X_train, y_train, X_test, y_test):

        # d = DataDLinear(X_train, y_train, X_test, y_test, lookback_len=self.params["lookback_len"], pred_len=self.params["pred_len"])
        # X_train, y_train, X_test, y_test = d.return_tensors()

        model = self.init_network()

        loss_fn = torch.nn.MSELoss()    # mean-squared error for regression
        loss_mae = loss = nn.L1Loss()

        optimiser = torch.optim.Adam(model.parameters(), lr=self.params["learning_rate"])
        batch_size = self.params["batch_size"]
        n_epochs= self.params["n_epochs"]


        for epoch in range(n_epochs):
            train_loss_l = []
            test_mse_loss_l = []
            test_mae_l = []
            r2_l = []

            metrics = []


            for b in range(0,len(X_train),batch_size):
                inpt = X_train[b:b+batch_size,:,:]
                target = y_train[b:b+batch_size]  


                model.train()
                outputs = model.forward(inpt) # forward pass
                optimiser.zero_grad() # calculate the gradient, manually setting to 0
                # obtain the loss function
                loss = loss_fn(outputs, target)
                loss.backward() # calculates the loss of the loss function
                optimiser.step() # improve from loss, i.e backprop
                train_loss_l.append(loss.item())
                # test loss
            for b in range(0,len(X_test),batch_size):
                inpt_test = X_test[b:b+batch_size,:,:]
                target_test = y_test[b:b+batch_size]  
                model.eval()
                test_preds = model(inpt_test)


                test_mse = loss_fn(test_preds, target_test)
                test_mae = loss_mae(test_preds, target_test)
                r2_ = r2_loss(test_preds, target_test)     # R2 can also be negative if the outputs are even worse than a function predicting the mean

                
                ''' [MSE, MAE, R2]'''
                metrics.append([test_mse.item(), test_mae.item(), r2_.item()])


                test_mse_loss_l.append(test_mse.item())
                test_mae_l.append(test_mae.item())
                
            
            metrics = np.array(metrics)
            metrics = np.mean(metrics, axis=0)

            # print("Test metrics [MSE, MAE, R2]: ", metrics)


            m_train_loss, m_test_loss, m_test_mae = np.mean(np.abs(np.array(train_loss_l))), np.mean(np.abs(np.array(test_mse_loss_l))),np.mean(np.abs(np.array(test_mae_l)))
            print("Epoch: %d, train loss: %1.5f, test loss: %1.5f,  test mae loss: %1.5f" % (epoch, 
                                                                        m_train_loss, 
                                                                        m_test_loss, 
                                                                        m_test_mae))
        
            
        return model, metrics



       

    