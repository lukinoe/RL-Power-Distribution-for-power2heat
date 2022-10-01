from flask import Flask
from pred import predict
from getData import ExtData
import pandas as pd
import numpy as np

extdata = ExtData()

app = Flask(__name__)

@app.route("/")
def hello_world():
    data = extdata.getDataFrame().values
    solar_forecast = data[:,0]
    load_forecast = data[:,1]

    print(solar_forecast,solar_forecast.shape)

    #predict(solar_forecast)
    res = predict(solar_forecast)

    print(res)
    print(res.shape)

    return res, load_forecast