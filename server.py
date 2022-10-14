from flask import Flask
from pred_thermal import predict_thermal
from pred_power import predict_power
from getData import ExtData
import pandas as pd
import numpy as np

extdata = ExtData()

app = Flask(__name__)

@app.route("/")
def predict():
    df, df_historic = extdata.getDataFrames()

    data = df.values
    solar_forecast = data[:,1]
    load_forecast = data[:,0]

    res_thermal = predict_thermal(df_historic[["i_temp1", "i_temp2", "i_temp3", "i_power", "i_m1sum"]], df["solar"])
    res_power = predict_power(df["solar"])

    print(res_thermal, res_power)
    print(res_thermal.shape, res_power.shape)

    response = {"load_forecast": list(load_forecast), 
                "solar_forecast": list(solar_forecast), 
                "predict_thermal": res_thermal.shape,
                "predict_power": list(res_power.flatten())
                }

    return response