import numpy as np


def find_runs(x):
    """Find runs of consecutive items in an array.

    Parameters
    ----------
    x : np.ndarray
        input array

    Returns
    -------
    run_values : np.ndarray
        values
    run_starts : np.ndarray
        indices of start of runs
    run_lengths : np.ndarray
        lengths of runs
    """
    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


def split_array_3(array, mid_size):
    """Randomly splits the array into 3 non-overlapping subsets.

    Parameters
    ----------
    array : array-like
        array to be splitted
    mid_size : int
        size of the middle subset

    Returns
    -------
    left : array-like
        left subset
    mid : array-like
        middle subset
    right : array-like
        right subset

    """
    array = np.array(array)
    n = len(array)
    if mid_size >= n:
        raise ValueError(f"{mid_size} >= {n}")

    m_start = np.random.choice(np.arange(n)[:-(mid_size + 1)])
    m_end = m_start + mid_size

    left_id = np.arange(0, m_start)
    mid_id = np.arange(m_start, m_end)
    right_id = np.arange(m_end, n)
    left = array[left_id]
    mid = array[mid_id]
    right = array[right_id]
    return left, mid, right