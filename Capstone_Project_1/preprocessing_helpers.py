# Capstone_Project_1/preprocessing_helpers.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class QuantileClipper(BaseEstimator, TransformerMixin):
    """
    Custom numeric column clipper used during training.
    Stores lower/upper quantiles during fit and clips values during transform.
    """
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99, columns=None):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.columns = columns
        self.lower_ = {}
        self.upper_ = {}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        cols = self.columns if self.columns else X.select_dtypes(include=[np.number]).columns.tolist()
        for c in cols:
            col = X[c].dropna()
            if col.shape[0] == 0:
                self.lower_[c], self.upper_[c] = 0.0, 0.0
            else:
                self.lower_[c] = float(col.quantile(self.lower_quantile))
                self.upper_[c] = float(col.quantile(self.upper_quantile))
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        Xc = X.copy()
        for c, low in self.lower_.items():
            high = self.upper_.get(c, low)
            if c in Xc.columns:
                try:
                    Xc[c] = pd.to_numeric(Xc[c], errors="coerce")
                    Xc[c] = Xc[c].clip(lower=low, upper=high)
                except Exception:
                    pass
        return Xc
