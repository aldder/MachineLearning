import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, drop_originals=True):
        self.drop_originals = drop_originals

    def fit(self, X, y=None):
        self.max = np.max(X, 0)
        if self.drop_originals:
            self.drop = np.arange(X.shape[1])
        return self

    def transform(self, X, y=None):
        X = np.array(X)
        for col in range(X.shape[1]):
            vmax = self.max[col]
            sin = np.sin(2 * np.pi * X[:, col] / vmax).reshape(-1, 1)
            cos = np.cos(2 * np.pi * X[:, col] / vmax).reshape(-1, 1)
            X = np.hstack([X, sin, cos])
        if self.drop_originals:
            X = np.delete(X, self.drop, axis=1)
        return X
