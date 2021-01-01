import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MAOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, window=336, scale=4):
        self.window = window
        self.scale = scale

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for c in X.columns:
            serie = X[c].dropna()
            if len(serie) <= self.window:
                continue
            rolling_mean = serie.rolling(window=self.window).mean()
            # confidence intervals for smoothed reg_values
            mae = np.mean(np.abs(serie[self.window:] - rolling_mean[self.window:]))
            deviation = np.std(serie[self.window:] - rolling_mean[self.window:])
            lower_bound = rolling_mean - (mae + deviation * self.scale)
            upper_bound = rolling_mean + (mae + deviation * self.scale)
            # Having the intervals, find abnormal values
            anomalies = pd.DataFrame(index=serie.index, columns=['anomaly'])
            anomalies[serie < lower_bound] = 1
            anomalies[serie > upper_bound] = 1
            X = pd.concat([X, anomalies], axis=1)
            X.loc[anomalies.dropna().index, c] = np.NaN
            X.drop(['anomaly'], 1, inplace=True)
        return X
