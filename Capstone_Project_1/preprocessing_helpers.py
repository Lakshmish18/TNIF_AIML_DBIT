# Capstone_Project_1/preprocessing_helpers.py
"""
Backward-compatible QuantileClipper.

This version is intentionally tolerant of older saved pickles that used
different internal attribute names (lower/upper, lower_quantile/upper_quantile,
lower_, lower_bounds_, etc.). It sets multiple attribute names on fit()
so older pickles and newer code both work.
"""

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class QuantileClipper(BaseEstimator, TransformerMixin):
    """
    Clips numeric features column-wise to the quantile range learned from the training data.

    Backwards compatibility notes:
    - older variants used attribute names like `lower`/`upper` or `lower_quantile`/`upper_quantile`
    - this implementation sets all of those names so unpickling older objects works.
    """

    def __init__(self, lower_quantile: float = None, upper_quantile: float = None, lower: float = None, upper: float = None):
        # Accept older param names as aliases
        if lower_quantile is None and lower is not None:
            lower_quantile = lower
        if upper_quantile is None and upper is not None:
            upper_quantile = upper

        self.lower_quantile = 0.01 if lower_quantile is None else float(lower_quantile)
        self.upper_quantile = 0.99 if upper_quantile is None else float(upper_quantile)

        # historic aliases
        self.lower = float(self.lower_quantile)
        self.upper = float(self.upper_quantile)

        # placeholders to be created in fit()
        self.lower_ = None
        self.upper_ = None
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)

        # store quantile values under multiple attribute names
        self.lower_ = float(self.lower_quantile)
        self.upper_ = float(self.upper_quantile)
        self.lower = self.lower_
        self.upper = self.upper_

        # compute bounds (nanquantile tolerates NaNs)
        try:
            lb = np.nanquantile(X_arr, self.lower_, axis=0)
            ub = np.nanquantile(X_arr, self.upper_, axis=0)
        except Exception:
            X2 = X_arr.reshape(-1, 1)
            lb = np.nanquantile(X2, self.lower_, axis=0)
            ub = np.nanquantile(X2, self.upper_, axis=0)

        # set bounds with multiple attribute names (compat)
        self.lower_bounds_ = np.asarray(lb, dtype=float)
        self.upper_bounds_ = np.asarray(ub, dtype=float)
        self.lower_bounds = self.lower_bounds_
        self.upper_bounds = self.upper_bounds_

        return self

    def transform(self, X):
        # If unpickled object used other attribute names, try to pick them up
        if getattr(self, "lower_bounds_", None) is None or getattr(self, "upper_bounds_", None) is None:
            lb = getattr(self, "lower_bounds", None)
            ub = getattr(self, "upper_bounds", None)
            if lb is None or ub is None:
                raise AttributeError("QuantileClipper appears not fitted: missing bounds.")
            self.lower_bounds_ = np.asarray(lb, dtype=float)
            self.upper_bounds_ = np.asarray(ub, dtype=float)

        X_arr = np.asarray(X, dtype=float).copy()
        lb = np.asarray(self.lower_bounds_, dtype=float)
        ub = np.asarray(self.upper_bounds_, dtype=float)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        # If bounds are scalar, broadcast
        if X_arr.shape[1] != lb.shape[0]:
            if lb.size == 1:
                lb = np.full((X_arr.shape[1],), lb.item())
            if ub.size == 1:
                ub = np.full((X_arr.shape[1],), ub.item())
            if X_arr.shape[1] != lb.shape[0]:
                raise ValueError(f"QuantileClipper: X has {X_arr.shape[1]} cols but bounds length is {lb.shape[0]}")

        X_clipped = np.maximum(X_arr, lb)
        X_clipped = np.minimum(X_clipped, ub)
        return X_clipped
