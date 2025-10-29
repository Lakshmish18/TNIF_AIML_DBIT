# Capstone_Project_1/preprocessing_helpers.py
"""
Compatibility helper for unpickling models that used a custom QuantileClipper.
This version aims to be backward-compatible with different saved versions.
"""
from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.01, upper: float = 0.99, lower_quantile: Optional[float] = None, upper_quantile: Optional[float] = None):
        if lower_quantile is not None:
            self.lower_quantile = float(lower_quantile)
        else:
            self.lower_quantile = float(lower)
        if upper_quantile is not None:
            self.upper_quantile = float(upper_quantile)
        else:
            self.upper_quantile = float(upper)
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        try:
            self.lower_bounds_ = np.nanquantile(X_arr, self.lower_quantile, axis=0)
            self.upper_bounds_ = np.nanquantile(X_arr, self.upper_quantile, axis=0)
        except Exception:
            x = X_arr.ravel()
            self.lower_bounds_ = np.nanquantile(x, self.lower_quantile)
            self.upper_bounds_ = np.nanquantile(x, self.upper_quantile)
        return self

    def transform(self, X):
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise ValueError("QuantileClipper is not fitted yet. Call fit(X) before transform.")
        X_arr = np.asarray(X, dtype=float).copy()
        lb = self.lower_bounds_
        ub = self.upper_bounds_
        try:
            X_arr = np.maximum(X_arr, lb)
            X_arr = np.minimum(X_arr, ub)
        except Exception:
            if X_arr.ndim == 1:
                return np.clip(X_arr, lb, ub)
            else:
                for col in range(X_arr.shape[1]):
                    lval = lb[col] if hasattr(lb, "__len__") and len(lb) > col else lb
                    uval = ub[col] if hasattr(ub, "__len__") and len(ub) > col else ub
                    X_arr[:, col] = np.clip(X_arr[:, col], lval, uval)
        return X_arr

    def __setstate__(self, state):
        if isinstance(state, dict):
            if "lower" in state and "lower_quantile" not in state:
                state["lower_quantile"] = state.pop("lower")
            if "upper" in state and "upper_quantile" not in state:
                state["upper_quantile"] = state.pop("upper")
            if "lower_" in state and "lower_bounds_" not in state:
                state["lower_bounds_"] = state.pop("lower_")
            if "upper_" in state and "upper_bounds_" not in state:
                state["upper_bounds_"] = state.pop("upper_")
        self.__dict__.update(state)
        if not hasattr(self, "lower_quantile"):
            self.lower_quantile = getattr(self, "lower", 0.01)
        if not hasattr(self, "upper_quantile"):
            self.upper_quantile = getattr(self, "upper", 0.99)
        if not hasattr(self, "lower_bounds_"):
            self.lower_bounds_ = None
        if not hasattr(self, "upper_bounds_"):
            self.upper_bounds_ = None

    def __getstate__(self):
        return {
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "lower_bounds_": self.lower_bounds_,
            "upper_bounds_": self.upper_bounds_,
        }
