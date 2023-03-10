from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
from torch import nn

from sklearn.utils.extmath import cartesian
from sklearn.preprocessing import OneHotEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


class DataLSTM():

    def __init__(self, X_train, y_train, X_test, y_test, lookback_len=100, pred_len=24) -> None:
        
        self.lookback_len = lookback_len
        self.pred_len = pred_len
        self.generate_sequences(X_train, y_train, X_test, y_test)
        

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

    def generate_sequences(self, X_train, y_train, X_test, y_test):
        print("***",X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        self.X_train, self.y_train = self.split_sequences(X_train, y_train)
        self.X_test, self.y_test = self.split_sequences(X_test, y_test)

        

    def return_tensors(self):

        X_train_tensors = Variable(torch.Tensor(self.X_train).to(device))
        X_test_tensors = Variable(torch.Tensor(self.X_test).to(device))

        y_train_tensors = Variable(torch.Tensor(self.y_train).to(device))
        y_test_tensors = Variable(torch.Tensor(self.y_test).to(device))


        X_train_tensors_final = torch.reshape(X_train_tensors,   
                                            (X_train_tensors.shape[0], self.lookback_len, 
                                            X_train_tensors.shape[2]))
        X_test_tensors_final = torch.reshape(X_test_tensors,  
                                            (X_test_tensors.shape[0], self.lookback_len, 
                                            X_test_tensors.shape[2])) 

        print("Training Shape:", X_train_tensors_final.shape, y_train_tensors.shape)
        print("Testing Shape:", X_test_tensors_final.shape, y_test_tensors.shape) 

        return X_train_tensors_final, y_train_tensors,X_test_tensors_final, y_test_tensors,


class LSTM(nn.Module):
    
    def __init__(self, out_len=100, input_size=48, hidden_size=100, seq_len=100, num_layers=1):
        super().__init__()
        self.out_len = out_len # output size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # neurons in each lstm layer
        self.seq_len = seq_len

        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.0) # lstm

        self.l_linear = torch.nn.Linear(self.hidden_size*self.seq_len, out_len)
        
    def forward(self,x):
        # hidden state
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device))
        # cell state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device))
        # propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # (input, hidden, and internal state)

        x = output.contiguous().view(x.size(0),-1)

        return self.l_linear(x)


class Trainer_LSTM():

    def __init__(self, params) -> None:

        self.params = params

    def init_network(self):

        lstm = LSTM(out_len=self.params["pred_len"], 
                    input_size=self.params["input_size"], 
                    hidden_size=self.params["hidden_size"], 
                    seq_len=self.params["lookback_len"],
                    num_layers=self.params["num_layers"])
        lstm.to(device)

        return lstm
        

    def training_loop(self, X_train, y_train, X_test, y_test):

        d = DataLSTM(X_train, y_train, X_test, y_test, lookback_len=self.params["lookback_len"], pred_len=self.params["pred_len"])
        X_train, y_train, X_test, y_test = d.return_tensors()

        lstm = self.init_network()

        loss_fn = torch.nn.MSELoss()    # mean-squared error for regression
        loss_mae = loss = nn.L1Loss()

        optimiser = torch.optim.Adam(lstm.parameters(), lr=self.params["learning_rate"])
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


                lstm.train()
                outputs = lstm.forward(inpt) # forward pass
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
                lstm.eval()
                test_preds = lstm(inpt_test)


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
        
            
        return lstm, metrics



       

    