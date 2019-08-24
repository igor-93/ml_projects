import numpy as np
from scipy.sparse import isspmatrix_coo, coo_matrix


def drop_data(mat, threshold):
    """Removes common from the matrix that is smaller then threshold.

    Parameters
    ----------
    mat : coo_matrix
        matrix
    threshold : float
         value below which we want to drop common

    Returns
    -------
    mat : coo
         matrix in the same format

    """
    if not isspmatrix_coo(mat):
        raise ValueError('Given matrix is not in COO format')

    mat.data[mat.data < threshold] = 0
    mat.eliminate_zeros()
    return mat


def drop_cols(mat, idx_to_drop):
    """Removes columns from the matrix.

    Parameters
    ----------
    mat : coo_matrix
        matrix
    idx_to_drop : array-like
        indices of columns we want to drop

    Returns
    -------
    mat: coo_matrix
        matrix without dropped columns
    """
    if not isspmatrix_coo(mat):
        raise ValueError('Given matrix is not in COO format')
    if np.max(idx_to_drop) >= mat.shape[1]:
        raise ValueError('Column indices are bigger then shape of the matrix.')

    idx_to_drop = np.unique(idx_to_drop)
    keep = ~np.in1d(mat.col, idx_to_drop)
    mat.data, mat.row, mat.col = mat.data[keep], mat.row[keep], mat.col[keep]
    mat.col -= idx_to_drop.searchsorted(mat.col)  # decrement column indices
    mat._shape = (mat.shape[0], mat.shape[1] - len(idx_to_drop))
    return coo_matrix(mat)


def drop_rows(mat, idx_to_drop):
    """Removes rows from the matrix.

    Parameters
    ----------
    mat : coo_matrix
        matrix
    idx_to_drop : array-like
        indices of rows we want to drop
    Returns
    -------
    mat: coo_matrix
        matrix without dropped rows
    """
    if not isspmatrix_coo(mat):
        raise ValueError('Given matrix is not in COO format')
    if not np.max(idx_to_drop) < mat.shape[0]:
        raise ValueError('Row indices are bigger then shape of the matrix.')

    idx_to_drop = np.unique(idx_to_drop)
    keep = ~np.in1d(mat.row, idx_to_drop)
    mat.data, mat.row, mat.col = mat.data[keep], mat.row[keep], mat.col[keep]
    mat.row -= idx_to_drop.searchsorted(mat.row)  # decrement row indices
    mat._shape = (mat.shape[0] - len(idx_to_drop), mat.shape[1])
    return coo_matrix(mat)


def make_zero_cols(mat, columns):
    """Annihilate entries in the given columns

    Parameters
    ----------
    mat : coo_matrix
        matrix
    columns : array-like
        indices of columns to set to 0

    Returns
    -------
    mat: coo_matrix
        matrix with given columns set to 0
    """
    if not isspmatrix_coo(mat):
        raise ValueError('Given matrix is not in COO format')
    if not np.max(columns) < mat.shape[1]:
        raise ValueError('Column indices are bigger then shape of the matrix.')

    columns = np.unique(columns)
    make_zero = np.in1d(mat.col, columns)
    mat.data[make_zero] = 0
    mat.eliminate_zeros()
    return mat.tocsr()
