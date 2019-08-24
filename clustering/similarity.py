import gc
from os import path
from enum import Enum
import itertools
import collections
import numpy as np
from scipy.sparse import csr_matrix, issparse, isspmatrix_csr, save_npz


# Supported similarity measures.
class SimMeasure(Enum):
    COSINE = 1
    JACCARD = 2
    JACCARD3 = 3
    JOHNSON = 4
    SIMPSON = 5

    def __str__(self):
        return self.name


def get_similarity(data, measure, intersection=None, is_distance=False, return_sparse=True, verbose=False):
    """

    Parameters
    ----------
    data
    measure
    intersection
    is_distance
    return_sparse

    Returns
    -------

    """
    n, d = data.shape

    if verbose:
        print('Building {} similarity for ({} x {}) matrix...'.format(measure, n, d))

    assert np.max(data) == 1 and np.min(data) == 0

    if verbose:
        print('Pre-computing row counts...')
    # Pre-compute nonzero counts for the rows
    if isspmatrix_csr(data):
        if verbose:
            print('Raw common is in CSR format.')
        nz_per_row = np.diff(data.indptr)
    else:
        nz_per_row = data.sum(axis=1)
        if isinstance(nz_per_row[0], collections.Sequence):
            nz_per_row = list(itertools.chain(*nz_per_row))

    data = csr_matrix(data)

    # Pre-compute intersection matrix
    if intersection is None:
        if verbose:
            print('Precomputing intersection...')
        intersection = np.dot(data, data.T)

    if issparse(intersection):
        if verbose:
            print('Converting intersection to dense...')
        intersection = intersection.todense()

    if verbose:
        print('Building similarity matrix...')
    if measure == SimMeasure.COSINE:
        intersection.astype(dtype=float, copy=False)
        b_prod_c = np.outer(nz_per_row, nz_per_row)
        b_prod_c = np.sqrt(b_prod_c)
        sim_matrix = np.divide(intersection, b_prod_c)
    elif measure == SimMeasure.JACCARD:
        b_add_c = np.add(nz_per_row[:, np.newaxis], nz_per_row[np.newaxis, :])
        sim_matrix = np.divide(intersection, np.subtract(b_add_c, intersection))
    elif measure == SimMeasure.JACCARD3:
        pure_b_c = np.subtract(nz_per_row[np.newaxis, :], intersection)
        b_intersection_c = np.add(pure_b_c, pure_b_c.T)
        sim_matrix = np.divide(3.0 * intersection, np.add(3.0 * intersection, b_intersection_c))
    elif measure == SimMeasure.JOHNSON:
        b_prod_c = 2.0 * np.outer(nz_per_row, nz_per_row)
        b_add_c = np.add(nz_per_row[:, np.newaxis], nz_per_row[np.newaxis, :])
        sim_matrix = np.divide(np.multiply(intersection, b_add_c), b_prod_c)
    elif measure == SimMeasure.SIMPSON:
        intersection.astype(dtype=float, copy=False)
        b_min_c = np.minimum(nz_per_row[:, np.newaxis], nz_per_row[np.newaxis, :])
        sim_matrix = np.divide(intersection, b_min_c)
    else:
        raise Exception('Similarity measure not supported: ', measure)

    sim_matrix *= 0.9999
    np.fill_diagonal(sim_matrix, 1.0)

    if verbose:
        print('Similarity matrix has shape: ', sim_matrix.shape)

    if is_distance:
        if verbose:
            print('Converting to distance matrix...')
        sim_matrix = 1 - sim_matrix
        if return_sparse:
            sim_matrix[sim_matrix == 1.0] = 0.0  # 0 is now for not connected points and ID
            sim_matrix = csr_matrix(sim_matrix)
    else:
        if return_sparse:
            sim_matrix = csr_matrix(sim_matrix)

    gc.collect()

    if return_sparse and verbose:
        occupancy = len(sim_matrix.data) / (sim_matrix.shape[0] * sim_matrix.shape[1]) * 100
        print('Similarity/Distance matrix has occupancy of %.4f percent' % occupancy)
    return sim_matrix
