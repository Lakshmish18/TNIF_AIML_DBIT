# preprocessing_helpers.py
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

class QuantileClipper(BaseEstimator, TransformerMixin):
    """
    Custom transformer that clips feature values to a given quantile range.
    Prevents extreme outliers from skewing model performance.
    """

    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X = np.asarray(X)
        # Compute per-column lower and upper quantiles
        self.lower_bounds_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X).copy()
        X = np.maximum(X, self.lower_bounds_)
        X = np.minimum(X, self.upper_bounds_)
        return X
