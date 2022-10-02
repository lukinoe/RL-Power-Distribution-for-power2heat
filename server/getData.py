from datetime import datetime
import asyncio
import requests
import json
import pandas as pd


def preprocess_load(res):

  data = json.loads(res)

  for i,e in enumerate(data):
    if e["name"]["en"] == "Hydro pumped storage consumption":
      x = e["xAxisValues"]
    if e["name"]["en"] == "Load forecast":
      y = e["data"]
      

  def get_date(x):
    digits = str(x)[:-3]
    return datetime.fromtimestamp(int(digits)).strftime("%Y-%m-%d %H:%M:%S")

  ts = [get_date(t) for t in x]
  df_load = pd.DataFrame({"date": ts, "val": y})
  df_load = df_load.set_index('date')
  df_load.index = pd.to_datetime(df_load.index)

  return df_load

  
class ExtData:
  def __init__(self, interval=100):

    self.interval = interval
    self.date_format = "%Y-%m-%d %H:%M:%S"


  def getDataFrame(self):
    self.get_data()
    df = self.merge()
    self.clean(df)

    return df


  def merge(self):
    df_date = pd.DataFrame({"date": pd.date_range(start = pd.datetime.today(), periods=self.interval, freq='15min')})
    df_date.date = df_date.date.round('15min')  
    df_date = df_date.set_index("date")

    merged_tmp = pd.merge(df_date,self.df_load, how='left', left_index=True, right_index=True)
    merged = pd.merge(merged_tmp,self.df_solar, how='left', left_index=True, right_index=True)
    merged.columns = ["load", "solar"]

    return merged


  def get_data(self):
    loop = asyncio.new_event_loop()
    self.df_solar = loop.run_until_complete(self.get_solar())
    self.df_load = loop.run_until_complete(self.get_load())


  @asyncio.coroutine
  def get_solar(self):
      loop = asyncio.get_event_loop()
      future1 = loop.run_in_executor(None, requests.get, 'https://api.forecast.solar/estimate/48.057194/14.346978/20/10/10')

      response1 = yield from future1
      res = response1.text
      data = json.loads(res)
      df_solar = pd.DataFrame(data["result"]["watts"], index=[0]).T
      df_solar.index = pd.to_datetime(df_solar.index)
      df_solar = pd.DataFrame(df_solar.iloc[:,0].resample('15min').ffill() / 4)

      return df_solar

  @asyncio.coroutine
  def get_load(self):
      loop = asyncio.get_event_loop()
      future1 = loop.run_in_executor(None, requests.get, 'https://www.energy-charts.info/charts/power/data/de/week_2022_39.json')

      response1 = yield from future1
      res = response1.text

      df = preprocess_load(res)
      return df

  def clean(self, df):
    df.solar = df.solar.fillna(0)
    df.load = df.load.fillna(0)

