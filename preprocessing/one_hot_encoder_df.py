import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator
from scipy import sparse


class OneHotEncoderDF(BaseEstimator):

    def __init__(self, categorical_feat_names, handle_unknown=None, sparse=None):
        self.categorical_feat_names = categorical_feat_names
        self.handle_unknown = handle_unknown
        self.sparse = sparse

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : pd.DataFrame
            input DataFrame
        y : None
        Returns
        -------
        """
        if self.categorical_feat_names is None:
            raise Exception("categorical_feat_names is not set.")
        self.categorical_feat_names = np.array(self.categorical_feat_names)
        feats_in = np.in1d(self.categorical_feat_names, X.columns)
        if not feats_in.all():
            raise ValueError("Missing features in data frame: ", self.categorical_feat_names[~feats_in])

        only_categorical_part = X[self.categorical_feat_names]

        self.encoder = OneHotEncoder(sparse=self.sparse, handle_unknown=self.handle_unknown)
        self.encoder.fit(only_categorical_part.values)
        return self

    def transform(self, X):
        feats_in = np.in1d(self.categorical_feat_names, X.columns)
        if not feats_in.all():
            raise ValueError("Missing features in data frame: ", self.categorical_feat_names[~feats_in])

        non_categorical = X.drop(columns=self.categorical_feat_names)
        transformed = self.encoder.transform(X[self.categorical_feat_names].values)
        if self.sparse:
            sparse_rest = sparse.csr_matrix(non_categorical.values)
            return sparse.hstack([transformed, sparse_rest])
        else:
            return np.hstack([transformed, non_categorical.values])
