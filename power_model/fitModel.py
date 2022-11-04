
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from utils.onehot_encode import onehot_build_dataset
import numpy as np
import pandas as pd
import math
import pickle
import os
_dir = os.path.dirname(os.path.realpath(__file__))


dataset = pd.read_csv(_dir + "/data/wout_m2sum_target_unscaled_date_m1_timefeatures_04-10.csv")
target = "power_consumption"


def fit_svr(dataset, encoding="none", scale=False, model="svr", epochs=200):
    
    
    X = dataset.iloc[:,:-1].values.astype(float)
    y = dataset.iloc[:,-1].values.astype(float).reshape(-1, 1)

    print("y mean and std:")
    print(np.mean(y), np.std(y))

    if encoding == "onehot":
      dataset = onehot_build_dataset(dataset, target)
      X = dataset[:,:-1]
      y = dataset[:,-1].reshape(-1, 1)

    sc_m1sum = StandardScaler()
    sc_y = StandardScaler()

    if scale:
      X[:,0] = sc_m1sum.fit_transform(X[:,0].reshape(-1, 1)).flatten()
      y = sc_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    print("y_test mean and std:")
    print(np.mean(y_test), np.std(y_test))

    # most important SVR parameter is Kernel type. It can be #linear,polynomial or gaussian SVR. We have a non-linear condition #so we can select polynomial or gaussian but here we select RBF(a #gaussian type) kernel.
    if model== "svr":
      regressor = SVR(kernel='rbf')
      regressor.fit(X_train,y_train)

      y_pred = regressor.predict(X_test)

      print("scaled MAE: ",mean_absolute_error(y_test, y_pred))
      if scale:
        y_test, y_pred = sc_y.inverse_transform(y_test.reshape(-1, 1)), sc_y.inverse_transform(y_pred.reshape(-1, 1))
      
    print(mean_absolute_error(y_test, y_pred))
    print("y_pred mean and std:")
    print(np.mean(y_pred), np.std(y_pred))


    return regressor, y_pred, y_test, sc_m1sum, sc_y


svr, y_pred, y_test, sc_m1sum, sc_y = fit_svr(dataset, scale=True, encoding="onehot")

# save model
with open(_dir + '/saved_models/model.pkl','wb') as f:
    pickle.dump(svr,f)

# save scaler
with open(_dir + '/saved_scaler/scaler_y.pkl','wb') as f:
    pickle.dump(sc_y,f)

with open(_dir + '/saved_scaler/scaler_m1sum.pkl','wb') as f:
    pickle.dump(sc_m1sum,f)


