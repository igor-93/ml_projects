import numpy as np
from sklearn.base import BaseEstimator


class SparseFeatureCompressor(BaseEstimator):

    def __init__(self, n_components=None):
        """Compresses sparse data to new dimension.
        TODO: add paper citation as source.
        Parameters
        ----------
        n_components: int
            new dimension
        """
        self.n_components = n_components
        self.phi = None

    def fit(self, X, y=None):
        # n in paper is actually dimension, so code will have the same naming convention as the paper

        n_samples, n = X.shape
        if self.n_components == n or self.n_components < 1:
            return self
        assert n > self.n_components

        # phi must have orthogonal columns
        phi = np.random.randn(n, self.n_components)
        den = (phi ** 2).sum(axis=0) ** 0.5
        phi = phi / den
        phi = gram_schmidt(phi)

        assert phi.shape == (n, self.n_components)

        self.phi = phi
        return self

    def transform(self, X):
        if self.n_components == X.shape[1] or self.n_components < 1:
            return X
        if self.phi is None:
            raise AssertionError('fit() was not called before transform()')
        new_X = np.dot(X, self.phi)
        return new_X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def gram_schmidt(X, row_vecs=False, norm=True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i, :].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i, :] - proj.sum(0)))
    if norm:
        Y = np.diag(1 / np.linalg.norm(Y, axis=1)).dot(Y)

    if row_vecs:
        return Y
    else:
        return Y.T