import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class SelectFeaturesByName(BaseEstimator):
    def __init__(self, best_features, feat_names=None):
        self.feat_names = feat_names
        self.best_features = best_features
        if feat_names is not None and not np.in1d(self.best_features, feat_names).all():
            raise ValueError("Missing features: ",
                             self.best_features[~np.in1d(self.best_features, feat_names)])
        self.mask = None

    def fit(self, X, y):
        if self.feat_names is None:
            if not y.index.equals(X.index):
                raise ValueError("indices mismatch")
            self.feat_names = X.columns
        if not np.in1d(self.best_features, self.feat_names).all():
            raise ValueError("Missing features: ",
                             self.best_features[~np.in1d(self.best_features, self.feat_names)])

        self.mask = np.in1d(self.feat_names, self.best_features)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            if not np.in1d(self.best_features, X.columns).all():
                raise ValueError("Missing features: ",
                                 self.best_features[~np.in1d(self.best_features, X.columns)])
            return X.loc[:, self.best_features]
        elif isinstance(X, np.ndarray):
            return X[:, self.mask]
        else:
            raise ValueError(f"input must be either ndarray or DataFrame, but it is {type(X)}")
