import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class InterpolationImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method):
        self.method = method
        assert self.method in ['linear','quadratic','cubic']

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for c in X.columns:
            X[c] = X[c].interpolate(method=self.method, limit_direction='both', fill_value='extrapolate')
        return X
