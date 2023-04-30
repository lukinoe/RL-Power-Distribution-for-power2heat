import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from datetime import date
import os

class DataSet:
    def __init__(self, start_date="2022-01-01", target="i_power", scale_target=False, scale_variables=False, time_features=False, dynamic_price=False, order=True, resample=None, demand_price=0.42, feedin_price=0.5) -> None:
        

        self.df = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + "/raw_2023.csv", sep=',', parse_dates={'date' : ['time']}, infer_datetime_format=True, index_col='date')
        self.start_date = start_date
        self.target = target
        self.scale_target = scale_target
        self.scale_variables = scale_variables
        self.time_features_ = time_features
        self.order = order
        self.scaler = StandardScaler()
        self.resample = resample

        self.dynamic_price = dynamic_price
        self.demand_price= demand_price
        self.feedin_price=feedin_price

        self.factor_temperature = 10
        self.factor_kwh = 1000

        self.order_df()

        self.transform_units()
        self.add_attributes()
        self.add_prices()

    def order_df(self):
        if self.order:
            self.df = self.df.iloc[::-1]  # reverse order: starting from lowest - to highest
            

    def transform_units(self):


        self.df["i_m1sum"] = self.df["i_m1sum"] / self.factor_kwh
        self.df["i_power"] = self.df["i_power"] / self.factor_kwh
        self.df["i_meterfeed"] = self.df["i_meterfeed"] / self.factor_kwh
        self.df["i_metercons"] = self.df["i_metercons"] / self.factor_kwh
        self.df["i_boostpower"] = self.df["i_boostpower"] / self.factor_kwh

        self.df["i_temp1"] = self.df["i_temp1"] / self.factor_temperature
        self.df["i_temp2"] = self.df["i_temp2"] / self.factor_temperature
        self.df["i_temp3"] = self.df["i_temp3"] / self.factor_temperature


    def add_attributes(self):

        '''
        Constants
        '''
        liter_in_tank = 500
        input_temperature = 30
        leistung_pro_grad = liter_in_tank * 4186 / 3600 / 1000 
        '''
        '''
        

        data = self.df.copy()

        power_consumption = data.i_m1sum - data.i_meterfeed - data.i_power - +data.i_metercons - data.i_boostpower #+ data.i_m2sum
        data["power_consumption_kwh"] = power_consumption.clip(lower=0)
        
        data["mean_temperature"] = (data.i_temp1 + data.i_temp2 + data.i_temp3) / 3 
        data["kwh_eq_state"] = (data.mean_temperature - input_temperature)*leistung_pro_grad
        data["kwh_eq_state"] = data["kwh_eq_state"].clip(lower=0)


        data["thermal_consumption_kwh"] = data["kwh_eq_state"].diff() - data["i_power"]
        data["thermal_consumption_kwh"] = data["thermal_consumption_kwh"].clip(upper=0)
        data["thermal_consumption_kwh"] = data["thermal_consumption_kwh"].abs()



        data["thermal_consumption"] = data["thermal_consumption_kwh"] / 1000
        data["power_consumption"] = data["power_consumption_kwh"] / 1000


        self.df = data

    def add_prices(self):

        data = self.df.copy()
        data["demand_price"] = self.demand_price
        data["feedin_price"] = self.feedin_price

        if self.dynamic_price: # arbitrary assumptions
            print("Dynamic prices: enabled")
            data.demand_price[(data.index.hour > 0) & (data.index.hour <= 6)  ] = 0.30
            data.demand_price[(data.index.hour > 6) & (data.index.hour <= 12)  ] = 0.45
            data.demand_price[(data.index.hour > 16) & (data.index.hour <= 19)  ] = 0.05
            data.demand_price[(data.index.hour > 12) & (data.index.hour <= 18)  ] = 0.47
            data.demand_price[(data.index.hour > 18) & (data.index.hour <= 22)  ] = 0.35
            data.demand_price[data.index.hour > 22] = 0.35


        self.df = data


    def time_features(self, data, return_te_only=False):
        data["date"] = pd.to_datetime(data.index)

        data["month"] = data["date"].dt.month
        data["day"] = data["date"].dt.day
        data["hour"] = data["date"].dt.hour
        data["weekday"] = data["date"].dt.dayofweek
        data["minute"] = data["date"].dt.minute


        del data["date"]

        if return_te_only:
            return data[["month",	"day",	"hour",	"weekday", "minute", self.target]]

        else:
            return data

    def scale_col(self, col):
        self.df[col] = self.scaler.fit_transform(self.df[col].values.reshape(-1, 1))

    def preprocess(self):
        self.df = self.df.fillna(0)
        self.df.sort_index()

    def resampler(self):
        self.df['date'] = pd.to_datetime(self.df.index)
        self.df = self.df.resample(self.resample, on='date').sum()
        if self.resample == "h":
            self.df.kwh_eq_state = self.df.kwh_eq_state / 4
            self.df.demand_price = self.df.demand_price / 4
            self.df.feedin_price = self.df.feedin_price / 4

    def pipeline(self):
        self.preprocess()
        self.df = self.df[(self.df.index > self.start_date)] # & (self.df.index <  "2022-08-19")]

        if self.time_features_:
            self.df = self.time_features(self.df)

        continuous_cols = ['i_power',
        'i_boostpower', 'i_meterfeed', 'i_metercons', 'i_temp1', 'i_m0l1',
        'i_m0l2', 'i_m0l3', 'i_m1sum', 'i_m1l1', 'i_m1l2', 'i_m1l3', 'i_m2sum',
        'i_m2l1', 'i_m2l2', 'i_m2l3', 'i_m2soc', 'i_m3sum', 'i_m3l1', 'i_m3l2',
        'i_m3l3', 'i_m3soc', 'i_m4sum', 'i_m4l1', 'i_m4l2', 'i_m4l3',
        'i_temp2', 'i_power1', 'i_power2', 'i_power3', 'i_temp3', 'i_temp4', "power_consumption", 	"thermal_consumption"	,"thermal_consumption_kwh"	, "power_consumption_kwh", "demand_price","energy_price"]

        continuous_cols.remove(self.target)

        if self.scale_variables:
            [self.scale_col(col) for col in continuous_cols]

        if self.scale_target:
            self.scale_col(self.target)

        if self.resample:
            self.resampler()
            

        self.df = self.df.reset_index()

        return self.df
    
        