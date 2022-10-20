from datetime import datetime
import asyncio
import mysql.connector
import requests
import json
import pandas as pd
import os

# Obtain connection string information from the portal
_dir = os.path.dirname(os.path.realpath(__file__))
f = open(_dir + '//config.json')
config = json.load(f)


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
  print(len(ts), len(y))
  if len(ts) == len(y):
    df_load = pd.DataFrame({"date": ts, "val": y})
  else: #if array length does not match --> return 0's
    df_load = pd.DataFrame({"date": ts, "val": [0]*len(ts)})  


  df_load = df_load.set_index('date')
  df_load.index = pd.to_datetime(df_load.index)

  return df_load

  
class ExtData:
  def __init__(self, interval=100):

    self.interval = interval
    self.date_format = "%Y-%m-%d %H:%M:%S"


  def getDataFrames(self):
    self.get_data()
    df = self.merge()
    self.clean(df)

    return df, self.df_historic


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
    self.df_historic = loop.run_until_complete(self.get_historic())


  @asyncio.coroutine
  def get_solar(self):
      loop = asyncio.get_event_loop()
      c = config["pv"]
      #https://api.forecast.solar/estimate/:lat/:lon/:dec/:az/:kwp

      future1 = loop.run_in_executor(None, requests.get, 'https://api.forecast.solar/estimate/{}/{}/{}/{}/{}'.format(c["lat"], c["lon"], c["declination"], c["azimuth"], c["kwhp"]))

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
      
      week = str(datetime.now().isocalendar().week)
      future1 = loop.run_in_executor(None, requests.get, 'https://www.energy-charts.info/charts/power/data/de/week_2022_' + week + '.json')

      response1 = yield from future1
      res = response1.text

      df = preprocess_load(res)
      return df

  @asyncio.coroutine
  def get_historic(self):
      try:
        conn = mysql.connector.connect(**config["db"])
        print("Connection established")
      except mysql.connector.Error as err:
          print(err)
      else:
        cursor = conn.cursor()

        # Read data
        cursor.execute("SELECT * FROM `my-pv-live`.teichstrasse9;")
        rows = cursor.fetchall()
        print("Read",cursor.rowcount,"row(s) of data.")

        df = pd.DataFrame(rows)
        df.columns = ["id", "date", "i_temp1", "i_temp2", "i_temp3", "i_power", "i_m1sum"]
        #print(df)

        conn.commit()
        cursor.close()
        conn.close()

        return df

  def clean(self, df):
    df.solar = df.solar.fillna(0)
    df.load = df.load.fillna(0)

