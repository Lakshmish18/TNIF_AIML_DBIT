"""
preprocessing_helpers.py

Robust QuantileClipper implementation compatible with different pickled pipeline variants.

This class exposes both attribute styles that older/newer pickles might expect:
 - lower_bounds_ / upper_bounds_
 - lower_ / upper_

That helps when unpickling model artifacts created with slightly different versions.
"""
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99, columns: Optional[List[str]] = None):
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)
        self.columns = columns

        # We'll populate both naming styles so unpickling works regardless of saved attribute names
        self.lower_bounds_ = None  # numpy array or dict
        self.upper_bounds_ = None
        self.lower_ = None
        self.upper_ = None

    def fit(self, X, y=None):
        # Accept DataFrame or 2D array
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            X = pd.DataFrame(X)

        if isinstance(X, pd.Series):
            X = X.to_frame()

        # Choose numeric columns by default
        cols = list(self.columns) if self.columns else X.select_dtypes(include=[np.number]).columns.tolist()

        # compute per-column quantiles and store in dict and arrays
        lower_dict: Dict[str, float] = {}
        upper_dict: Dict[str, float] = {}

        for c in cols:
            if c not in X.columns or X[c].dropna().shape[0] == 0:
                lower_dict[c] = 0.0
                upper_dict[c] = 0.0
            else:
                col = pd.to_numeric(X[c], errors="coerce").dropna()
                lower_dict[c] = float(np.nanquantile(col, self.lower_quantile))
                upper_dict[c] = float(np.nanquantile(col, self.upper_quantile))

        # store as dicts
        self.lower_ = lower_dict.copy()
        self.upper_ = upper_dict.copy()

        # also store as numpy arrays and aligned list for compatibility if caller expects arrays
        cols_sorted = cols
        self._cols_order = cols_sorted
        self.lower_bounds_ = np.array([lower_dict[c] for c in cols_sorted], dtype=float)
        self.upper_bounds_ = np.array([upper_dict[c] for c in cols_sorted], dtype=float)

        return self

    def transform(self, X):
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            X = pd.DataFrame(X)

        if isinstance(X, pd.Series):
            X = X.to_frame()

        Xc = X.copy()

        # If we have dict-style bounds, apply them column-by-column
        if isinstance(self.lower_, dict) and len(self.lower_) > 0:
            for c, low in self.lower_.items():
                high = self.upper_.get(c, low)
                if c in Xc.columns:
                    try:
                        Xc[c] = pd.to_numeric(Xc[c], errors="coerce")
                        Xc[c] = Xc[c].clip(lower=low, upper=high)
                    except Exception:
                        # If casting fails, skip quietly
                        pass
        else:
            # fallback: apply array-style clipping for numeric columns in stored order
            if getattr(self, "_cols_order", None) is None:
                return Xc
            for idx, c in enumerate(self._cols_order):
                low = float(self.lower_bounds_[idx])
                high = float(self.upper_bounds_[idx])
                if c in Xc.columns:
                    try:
                        Xc[c] = pd.to_numeric(Xc[c], errors="coerce")
                        Xc[c] = Xc[c].clip(lower=low, upper=high)
                    except Exception:
                        pass
        return Xc
