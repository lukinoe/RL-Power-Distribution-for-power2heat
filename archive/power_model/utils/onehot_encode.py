from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


def add_time_features(data, mode="power"):
    data = data.reset_index()
    data["date"] = pd.to_datetime(data.date)

    if mode=="thermal":
      data['month'] = data.date.apply(lambda row: row.month, 1)
      data['day'] = data.date.apply(lambda row: row.day, 1)
      data['weekday'] = data.date.apply(lambda row: row.weekday(), 1)
      data['hour'] = data.date.apply(lambda row: row.hour, 1)
      #data['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)

    else: # power
      data["month"] = data["date"].dt.month
      data["day"] = data["date"].dt.day
      data["hour"] = data["date"].dt.hour
      data["weekday"] = data["date"].dt.dayofweek
      data["minute"] = data["date"].dt.minute


    del data["date"]

    return data

def cats(start, end):
  a = np.array([str(i) for i in range(start, end+1)])
  return list(a.reshape(1,len(a)))

def encode_minute():
  return list(np.array(["0", "15", "30", "45"]).reshape(1,4))

def encode_and_bind(original_dataframe, feature_to_encode, categories):
    original_dataframe = original_dataframe.astype(str)  # pd dummies only works for str input
    print(categories)
    enc = OneHotEncoder(categories=categories)

    dummies = enc.fit_transform(original_dataframe[[feature_to_encode]]).toarray()
    dummies = pd.DataFrame(dummies)
    res = pd.concat([original_dataframe, dummies], axis=1)
    del res[feature_to_encode]
    return res


def onehot_build_dataset(data, target, del_target=True):
  df = data
  for i in [["day", cats(1,31)] , 
            ["month", cats(1,12)], 
            [ "hour", cats(0,23)], 
            [ "weekday", cats(0,6)] , 
            [ "minute" , encode_minute()]]:

    df = encode_and_bind(df, i[0], i[1])

  if del_target:
    del df[target]
    df[target] = data[target]


  dataset = df.to_numpy(dtype=float)
  print("dataset shape", dataset.shape)

  return dataset