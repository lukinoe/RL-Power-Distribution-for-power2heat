import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#2 Importing the dataset
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn import ensemble
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from data_utils import onehot_build_dataset


class Model:


    def __init__(self, dataset, encoding, scale, model="svr", target="power_consumption", test_size=0.05, epochs=100) -> None:
      self.dataset = dataset
      self.encoding = encoding 
      self.scale = scale 
      self.model = model
      self.target = target
      self.test_size = test_size
      self.epochs = epochs

      if scale:
        self.sc_y = StandardScaler()

    def train(self, model="svr"):

        model = self.model
        dataset = self.dataset
        epochs = self.epochs
        
        X = dataset.iloc[:,:-1].values.astype(float)
        y = dataset.iloc[:,-1].values.astype(float).reshape(-1, 1)

        print("y mean and std:")
        print(np.mean(y), np.std(y))

        if self.encoding == "onehot":
            dataset = onehot_build_dataset(dataset, self.target)
            X = dataset[:,:-1]
            y = dataset[:,-1].reshape(-1, 1)


        if self.scale:
            y = self.sc_y.fit_transform(y)


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42, shuffle=False)

        print("y_test mean and std:")
        print(np.mean(y_test), np.std(y_test))

        # most important SVR parameter is Kernel type. It can be #linear,polynomial or gaussian SVR. We have a non-linear condition #so we can select polynomial or gaussian but here we select RBF(a #gaussian type) kernel.
        if model== "svr":
            regressor = SVR(kernel='rbf')

            regressor.fit(X_train,y_train)

            y_pred = regressor.predict(X_test)

        if model == "nn":
            train_data = tf.data.Dataset.from_tensors((X_train, y_train))
            valid_data = tf.data.Dataset.from_tensors((X_test, y_test))


            model = keras.Sequential()
            model.add(layers.Dense(1000,  activation="relu"))
            model.add(layers.Dense(200,  activation="sigmoid"))
            model.add(layers.Dense(1, activation="linear"))
            model.compile(loss="mean_squared_error", optimizer="adam", metrics=[[tf.keras.metrics.MeanAbsoluteError()]])
            
            H = model.fit(train_data, validation_data=valid_data, epochs=epochs, batch_size=64)
            model.summary()

            regressor = model
            y_pred = model.predict(tf.data.Dataset.from_tensors(X_test))


        return regressor, y_pred, y_test

    def results(self, plot=True):

        regressor, y_pred, y_test = self.train()

        print("scaled MAE: ", mean_absolute_error(y_test, y_pred))
        if self.scale:
            y_test, y_pred = self.sc_y.inverse_transform(
            y_test.reshape(-1, 1)), self.sc_y.inverse_transform(y_pred.reshape(-1, 1))


        print(mean_absolute_error(y_test, y_pred))
        
        print("y_pred mean and std:")
        print(np.mean(y_pred), np.std(y_pred))


        if plot:
            plt.plot(y_pred[:500])
            plt.plot(y_test[:500])
            plt.show()

