import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#2 Importing the dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn import ensemble
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping

from lstm import DataLSTM, LSTM, Trainer_LSTM



from data_utils import onehot_build_dataset


class Model:


    def __init__(self, dataset, encoding, scale, model="svr", target="power_consumption", test_size=0.05, epochs=200, verbose=1, 

        nn_params={'activation1': 'sigmoid', 'activation2': 'selu', 'lr': 0.001, 'n_hidden1': 500, 'n_hidden2': 50},
        svr_params={"kernel": "rbf"},
        lstm_params={"n_epochs": 20, "learning_rate":0.001, "batch_size": 64 , "hidden_size": 50, "num_layers": 1, "lookback_len": 100, "pred_len": 24}

    ) -> None:
      self.dataset = dataset
      self.encoding = encoding 
      self.scale = scale 
      self.model = model
      self.target = target
      self.test_size = test_size
      self.epochs = epochs
      self.verbose = verbose
      self.nn_params = nn_params
      self.svr_params = svr_params
      self.lstm_params = lstm_params


      if scale:
        self.sc_y = StandardScaler()

    def prepare_data(self):
        dataset = self.dataset
        
        X = dataset.iloc[:,:-1].values.astype(float)
        if self.model== "lstm":
            X = dataset.values.astype(float)

        y = dataset.iloc[:,-1].values.astype(float).reshape(-1, 1)

        if self.encoding == "onehot":
            dataset = onehot_build_dataset(dataset, self.target)

            X = dataset[:,:-1]
            if self.model== "lstm":
                X = dataset

            y = dataset[:,-1].reshape(-1, 1)


        if self.scale:
            y = self.sc_y.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42, shuffle=False)

        return X_train, X_test, y_train, y_test


    def train(self, model="svr"):


        X_train, X_test, y_train, y_test = self.prepare_data()

        if self.model== "svr":
            regressor, y_pred = self.train_model_svr(X_train, y_train, X_test, y_test)

        if self.model == "nn":
            regressor, y_pred = self.train_model_nn(X_train, y_train, X_test, y_test)

        if self.model == "lstm":
            regressor, y_pred = self.train_model_lstm(X_train, y_train, X_test, y_test)

        return regressor, y_pred, y_test

    def results(self, plot=True):

        regressor, y_pred, y_test = self.train()

        if self.model != "lstm":

            print("scaled MAE: ", mean_absolute_error(y_test, y_pred))
            
            if self.scale:
                y_test, y_pred = self.sc_y.inverse_transform(
                y_test.reshape(-1, 1)), self.sc_y.inverse_transform(y_pred.reshape(-1, 1))
            
            print("y_pred mean and std:")
            print(np.mean(y_pred), np.std(y_pred))

            if plot:
                plt.plot(y_pred[:500])
                plt.plot(y_test[:500])
                plt.show()

            return self.get_metrics(y_pred, y_test)

        else:
            metrics_dict = {"mae": y_pred}

        return metrics_dict


    def train_model_nn(self, X_train, y_train, X_test, y_test):
        train_data = tf.data.Dataset.from_tensors((X_train, y_train))
        valid_data = tf.data.Dataset.from_tensors((X_test, y_test))

        early_stopping_monitor = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=25,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True
        )
        initializer = tf.keras.initializers.GlorotNormal()  # = Xavier Init
        #optimizer = keras.optimizers.Adam(lr=self.nn_params.lr)

        model = keras.Sequential()
        model.add(layers.Dense(self.nn_params["n_hidden1"],  activation=self.nn_params["activation1"], kernel_initializer=initializer))
        model.add(layers.Dense(self.nn_params["n_hidden2"],  activation=self.nn_params["activation2"], kernel_initializer=initializer))
        model.add(layers.Dense(1, activation="linear", kernel_initializer=initializer))
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=[
            [tf.keras.metrics.MeanAbsoluteError()]])


        H = model.fit(train_data, validation_data=valid_data, epochs=self.epochs,
                    batch_size=64, callbacks=[early_stopping_monitor], verbose=self.verbose)
        model.summary()

        regressor = model
        y_pred = model.predict(tf.data.Dataset.from_tensors(X_test))

        return regressor, y_pred

    def train_model_svr(self, X_train, y_train, X_test, y_test):
        regressor = SVR(kernel=self.svr_params["kernel"], C=self.svr_params["C"], epsilon=self.svr_params["epsilon"])
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)

        return regressor, y_pred

    def train_model_lstm(self, X_train, y_train, X_test, y_test):
        print(X_train.shape, y_test.shape)

        self.lstm_params.update({"input_size": X_train.shape[1]})
        trainer = Trainer_LSTM(self.lstm_params)
        regressor, y_pred = trainer.training_loop(X_train, y_train, X_test, y_test)

        return regressor, y_pred
        

    
    def get_metrics(self, y_pred, y_test):

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)


        return  {"mae": mae, "mse": mse, "mape":mape, "r2":r2 }

