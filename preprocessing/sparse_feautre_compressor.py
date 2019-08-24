import numpy as np
from sklearn.base import BaseEstimator


class SparseFeatureCompressor(BaseEstimator):

    def __init__(self, new_dim):
        """Compresses sparse data to new dimension.
        TODO: add paper citation as source.

        Parameters
        ----------
        new_dim: int
            new dimension
        """
        self.m = new_dim
        self.phi = None

    def fit(self, X, y=None):
        # n in paper is actually dimension, so code will have the same naming convention as the paper

        n_samples, n = X.shape
        assert n > self.m

        # phi must have orthogonal columns
        phi = np.random.randn(n, self.m)
        den = (phi ** 2).sum(axis=0) ** 0.5
        phi = phi / den
        phi = gram_schmidt(phi)

        assert phi.shape == (n, self.m)

        self.phi = phi
        return self

    def transform(self, X):
        if self.phi is None:
            raise AssertionError('fit() was not called before transform()')
        new_X = np.dot(X, self.phi)
        return new_X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
