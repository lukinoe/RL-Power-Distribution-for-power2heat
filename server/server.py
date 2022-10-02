from flask import Flask
from pred_thermal import predict_thermal
from power_model.pred_power import predict_power
from getData import ExtData
import pandas as pd
import numpy as np

extdata = ExtData()

app = Flask(__name__)

@app.route("/")
def predict():
    df = extdata.getDataFrame()

    data = df.values
    solar_forecast = data[:,1]
    load_forecast = data[:,0]

    #predict(solar_forecast)
    res_thermal = predict_thermal(solar_forecast)
    res_power = predict_power(df["solar"])


    print(res_thermal, res_power)
    print(res_thermal.shape, res_power.shape)

    response = {"load_forecast": list(load_forecast), 
                "solar_forecast": list(solar_forecast), 
                "predict_power": list(res_power.flatten()), 
                "predict_thermal": res_thermal.shape}

    return response