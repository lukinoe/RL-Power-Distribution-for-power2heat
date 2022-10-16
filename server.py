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

    res_thermal = predict_thermal(df_historic[["i_m1sum","i_power","i_temp1","i_temp2","i_temp3"]], df["solar"])
    res_power = predict_power(df["solar"])

    response = {"load_forecast": list(load_forecast), 
                "solar_forecast": list(solar_forecast), 
                "historical_temp": df_historic[["i_temp1","i_temp2","i_temp3"]].values.tolist(),
                "predict_temp": res_thermal.tolist(),
                "predict_power": res_power.flatten().tolist()
                }

    return response