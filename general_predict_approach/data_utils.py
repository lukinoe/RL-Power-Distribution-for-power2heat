import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import math
from itertools import combinations
import torch
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cats(start, end):
  a = np.array([str(i) for i in range(start, end+1)])
  return list(a.reshape(1,len(a)))

def encode_minute():
  return list(np.array(["0", "15", "30", "45"]).reshape(1,4))


def encode_and_bind(original_dataframe, feature_to_encode, categories):
    original_dataframe = original_dataframe.astype(str)  # pd dummies only works for str input
    enc = OneHotEncoder(categories=categories)

    dummies = enc.fit_transform(original_dataframe[[feature_to_encode]]).toarray()
    dummies = pd.DataFrame(dummies)
    res = pd.concat([original_dataframe, dummies], axis=1)
    del res[feature_to_encode]
    return res


def onehot_build_dataset(data, target):
  df = data.copy()

  for i in [["day", cats(1,31)] , 
            ["month", cats(1,12)], 
            [ "hour", cats(0,23)], 
            [ "weekday", cats(0,6)] , 
            [ "minute" , encode_minute()]]:


    #df = encode_and_bind(df, i[0])
    if i[0] in data.columns:
      df = encode_and_bind(df, i[0], i[1])


  # cols = df.columns.tolist()
  # cols = cols[-1:] + cols[:-1]
  # df = df[cols]
  # df.insert(-1, target, df.pop(target))

  del df[target]
  df[target] = data[target]
  dataset = df.to_numpy(dtype=float)
  print("dataset shape", dataset.shape)
  return dataset

def cyclical_transform(df, col, del_old=False):
  max_value = df[col].max()
  sin_values = [math.sin((2*math.pi*x)/max_value) for x in list(df[col])]
  cos_values = [math.cos((2*math.pi*x)/max_value) for x in list(df[col])]
  df[col + "_sin"] = sin_values
  df[col + "_cos"] = cos_values
  if del_old:
    del df[col]
  return df  

def cyclical_encode_dataset(data, target):

  df = data.copy()

  for i in ["day", "month", "hour", "weekday", "minute"]:
    if i in df.columns:
      df = cyclical_transform(df, i, True)

  del df[target]
  #del df["date"]
  df[target] = data[target]

  dataset = df.to_numpy(dtype=float)

  return dataset, df


def get_feature_combinations(features, target):

  features = features[:-1]
  feature_combinations = []
  for i in range(len(features)):
      oc = combinations(features, i + 1)
      for c in oc:
          l = list(c)
          l.append(target)
          feature_combinations.append(l)

  return feature_combinations



class TimeSeriesTransform():

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


