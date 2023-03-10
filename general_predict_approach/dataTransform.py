
import pandas as pd
from sklearn.preprocessing import StandardScaler



class Transform:
    def __init__(self, resample="h", dataset=None, target="power_consumption", scale_X=True) -> None:
        self.resample = resample
        self.target = target
        self.scale_X = scale_X
        self.sc_X = StandardScaler()
        
        self.dataset = dataset


    def transform(self):

        dataset = self.dataset
        target = self.target

        d = dataset.copy()

        d['date'] = pd.to_datetime(d['date'])
        d = d.resample(self.resample, on='date').sum()

        val_last_day = []
        mean_24_hours = []
        val_last_week = []
        for i in range(len(d)):
            if i < 24:
                val_last_day.append(d[target].mean())
                mean_24_hours.append(d[target].mean())
                val_last_week.append(d[target].mean())
            else:
                val_last_day.append(d[target].iloc[i-24])
                val_last_week.append(d[target].iloc[i-(24*7)])
                mean_24_hours.append(d[target].iloc[i-24:i].mean())
                

        d["val_last_day"] = val_last_day
        d["val_last_week"] = val_last_week
        d["mean_24h"] = mean_24_hours


        d["month"] = d.index.month
        d["hour"] = d.index.hour
        d["weekday"] = d.index.dayofweek
        d["day_continuous"] = d.index.day
        _d = d[target]
        del d[target]
        d[target] = _d

        d = d.reset_index()
        del d["date"]

        if self.scale_X:
            for col in d.columns:
                if col not in ["month", "hour", "weekday", "day", target]:
                    d[col] = self.sc_X.fit_transform(d[col].values.reshape(-1, 1))

        return d

