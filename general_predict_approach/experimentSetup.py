import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.api import OLS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.svm import SVR
from models.LSTM import Trainer_LSTM
from models.DLinear import Trainer_DLinear
from models.FFN import FFN

from models.DLinear import DataDLinear

from data_utils import TimeSeriesTransform, onehot_build_dataset, cyclical_encode_dataset


class Model:


    def __init__(self, dataset, encoding, scale, model="svr", target="power_consumption", test_size=0.05, shuffle=False, epochs=200, verbose=1, 

        # model_params={"n_epochs": 20, "learning_rate":0.001, "batch_size": 64, 'activation1': 'sigmoid', 'activation2': 'selu', 'lr': 0.001, 'n_hidden1': 500, 'n_hidden2': 50, "hidden_size": 50, "num_layers": 1, "lookback_len": 100, "pred_len": 24, "kernel": "rbf", "C": 1,"epsilon": 0.1},
        model_params={},

    ) -> None:
      self.dataset = dataset
      self.encoding = encoding 
      self.scale = scale 
      self.model = model
      self.target = target
      self.test_size = test_size
      self.shuffle = shuffle
      self.epochs = epochs
      self.verbose = verbose

      self.model_params = model_params
  

      if scale:
        self.sc_y = StandardScaler()

    def prepare_data(self):
        dataset = self.dataset

        timeseries = False
        if self.model in ["lstm", "DLinear"]: 
            timeseries = True
        
        if timeseries:
            X = dataset.values.astype(float)

        else:
            X = dataset.iloc[:,:-1].values.astype(float)

        y = dataset.iloc[:,-1].values.astype(float).reshape(-1, 1)

        if self.encoding == "onehot":

            dataset = onehot_build_dataset(dataset, self.target)

            X = dataset[:,:-1]

            if timeseries:
                X = dataset

            y = dataset[:,-1].reshape(-1, 1)

        if self.encoding == "cyclical":
            dataset, _ = cyclical_encode_dataset(dataset, self.target)

            X = dataset[:,:-1]

            if timeseries:
                X = dataset

            y = dataset[:,-1].reshape(-1, 1)

        if self.scale:
            y = self.sc_y.fit_transform(y)

        

        if self.model == "DLinear": # Timerseries preprocessing
            d = TimeSeriesTransform(X,y, lookback_len=self.model_params["lookback_len"], pred_len=self.model_params["pred_len"])
            X,y = d.return_tensors()
            

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42, shuffle=self.shuffle)
        print("SHapezzz", X_train.shape, X_test.shape, y_train.shape, y_test.shape )

        return X_train, X_test, y_train, y_test


    def train(self, model="svr"):


        X_train, X_test, y_train, y_test = self.prepare_data()

        if self.model =="linear_regression":
            regressor, y_pred = self.train_model_LinearRegression(X_train, y_train, X_test, y_test)

        if self.model== "svr":
            regressor, y_pred = self.train_model_svr(X_train, y_train, X_test, y_test)

        if self.model == "nn":
            regressor, y_pred = self.train_model_nn(X_train, y_train, X_test, y_test)

        if self.model == "lstm":
            regressor, y_pred = self.train_model_lstm(X_train, y_train, X_test, y_test)

        if self.model == "DLinear":
            regressor, y_pred = self.train_model_DLinear(X_train, y_train, X_test, y_test)

            
            
        return regressor, y_pred, y_test

    def results(self, plot=True):

        regressor, y_pred, y_test = self.train()

        if self.model not in ["lstm", "DLinear"]:

            metrics_dict = self.get_metrics(y_pred, y_test)

            print("scaled MAE: ", mean_absolute_error(y_test, y_pred))
            
            """ ATTENTION: the result dict for the benchmarks is not inverse scaled!"""

            
            if self.scale:
                y_test, y_pred = self.sc_y.inverse_transform(y_test.reshape(-1, 1)), self.sc_y.inverse_transform(y_pred.reshape(-1, 1))
            
            print("y_pred mean and std:")
            print(np.mean(y_pred), np.std(y_pred))

            if plot:
                plt.plot(y_pred[:500], label="Prediction")
                plt.plot(y_test[:500], label="real")
                plt.xlabel('timesteps')
                plt.ylabel('kwH')
                plt.legend()
                plt.show()


        else:
            metrics_dict = {"mae": y_pred[1], 
                            "mse": y_pred[0], 
                            "mape": y_pred[2], 
                            "r2": y_pred[3]}

        return metrics_dict
    


    def train_model_LinearRegression(self, X_train, y_train, X_test, y_test):
            
        regressor = OLS(y_train, X_train).fit()
        y_pred = regressor.predict(X_test)

        return regressor, y_pred
    
    
    def train_model_svr(self, X_train, y_train, X_test, y_test):
        regressor = SVR(kernel=self.model_params["kernel"], C=self.model_params["C"], epsilon=self.model_params["epsilon"])
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)

        return regressor, y_pred


    def train_model_nn(self, X_train, y_train, X_test, y_test):

        model = FFN(self.model_params)
        regressor, y_pred = model.train(X_train, y_train, X_test, y_test)

        return regressor, y_pred


    def train_model_lstm(self, X_train, y_train, X_test, y_test):
        print(X_train.shape, y_test.shape)

        self.model_params.update({"input_size": X_train.shape[1]})
        print(self.model_params)
        trainer = Trainer_LSTM(self.model_params)
        regressor, y_pred = trainer.training_loop(X_train, y_train, X_test, y_test)

        return regressor, y_pred
    

    def train_model_DLinear(self, X_train, y_train, X_test, y_test):
        print(X_train.shape, y_test.shape)

        self.model_params.update({"input_size": X_train.shape[1]})
        print(self.model_params)
        trainer = Trainer_DLinear(self.model_params)
        regressor, y_pred = trainer.training_loop(X_train, y_train, X_test, y_test)

        return regressor, y_pred
        

    
    def get_metrics(self, y_pred, y_test):

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        return  {"mae": mae, "mse": mse, "mape":mape, "r2":r2 }


