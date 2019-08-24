import numpy as np
from numba import jit


@jit(nopython=True)
def ewma_span(x, span=None):
    """Returns the exponentially weighted moving average of x.

    Parameters
    ----------
    x : array-like
        input array
    alpha : float, optional
        value between 0 and 1
    span : float, optional
        similar to window size in rolling mean

    Returns
    -------
    ewma : numpy array
        the exponentially weighted moving average

    Example:
    --------
        n = 120
        span = 120
        rev_alpha = 1.0 - (2.0 / (span+1.0))
        exponents = np.arange(n-1, -1, -1)
        weights = [rev_alpha ** e for e in exponents]
        plt.plot(weights)

    """
    alpha = 2.0 / (span + 1.0)

    n = len(x)
    ewa = [x[0]]
    for i in range(1, n):
        e = alpha * x[i] + (1 - alpha) * ewa[i - 1]
        ewa.append(e)

    assert len(ewa) == n

    return np.array(ewa)


def ewma(x, alpha=None, span=None):
    """Returns the exponentially weighted moving average of x.

    Parameters
    ----------
    x : array-like
        input array
    alpha : float, optional
        value between 0 and 1
    span : float, optional
        similar to window size in rolling mean

    Returns
    -------
    ewma : numpy array
        the exponentially weighted moving average

    Example:
    --------
        n = 120
        span = 120
        rev_alpha = 1.0 - (2.0 / (span+1.0))
        exponents = np.arange(n-1, -1, -1)
        weights = [rev_alpha ** e for e in exponents]
        plt.plot(weights)

    """
    if alpha is None and span is None:
        raise ValueError('At span or alpha has to be given.')
    if alpha is not None and span is not None:
        raise ValueError('Either give alpha or span, not both.')
    if span is not None:
        alpha = 2.0 / (span + 1.0)
    if alpha is not None:
        if not (0.0 < alpha < 1.0):
            raise ValueError('Alpha in wrong range: ', alpha)
    # coerce x to an array
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    x = np.squeeze(x)
    if len(x.shape) != 1:
        raise ValueError('Must be 1-D, but it is: ', x.shape, x.size)
    n = x.size
    # create an initial weight matrix of (1-alpha), and a matrix of powers
    # to raise the weights by
    w0 = np.ones(shape=(n, n)) * (1 - alpha)
    p = np.vstack([np.arange(i, i - n, -1) for i in range(n)])
    # create the weight matrix
    w = np.tril(w0 ** p, 0)
    # calculate the ewma
    return np.dot(w, x[::np.newaxis]) / w.sum(axis=1)
