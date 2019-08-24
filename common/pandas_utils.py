import warnings
import numpy as np
import pandas as pd
from numba import jit


@jit(nopython=True)
def rolling_multi_columns_helper(array, window, fn):
    """Helper function for rolling multiple columns. Interval is closed on the right and the labeling is right

    Parameters
    ----------
    array : np.array
        array on wich we perform rolling
    window : int
        window size
    fn : callable
        it reduces 1d or 2d (depends on array shape) array to scalar

    Returns
    -------
    res : np.array
        result
    """
    n = len(array)
    res = np.empty(array.shape[0])
    res[:] = np.nan
    for i in np.arange(window, n + 1):
        curr = array[i - window:i]
        r = fn(curr)
        res[i - 1] = r

    return res


def rolling_multi_columns(df, columns, window, fn):
    """Function for rolling multiple columns. Interval is closed on the right and the labeling is right.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe
    columns : `list`of `str`
        list of columns
    window : int
        window size
    fn : callable
        it reduces 1d or 2d (depends on array shape) array to scalar

    Returns
    -------
    result : pd.Series
        results

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(df)
    res = rolling_multi_columns_helper(df[columns].values, window, fn)
    result = pd.Series(res, index=df.index)
    return result


def column_name_to_id(df_columns, *names):
    """For given list of columns and wanted names, the function returns the ids of wanted columns. These ids can
    then be used with .iloc

    Parameters
    ----------
    df_columns : [str]
        list of columns in the DataFrame.
    names : str
        list of names that will be searched for. If they do not exist, their id is set to -1

    Returns
    -------
    ids : [int]
        list of indices of wanted columns
    """
    df_columns = np.array(df_columns)
    result = []
    for name in names:
        col_ids = []
        for c in name:
            try:
                cur_id = np.where(df_columns == c)[0][0]
            except IndexError:
                warnings.warn('Column {} not found'.format(c))
                cur_id = -1
            col_ids.append(cur_id)

        result.append(col_ids)

    return result


def intersect(*data):
    """The function returns the intersection of all the data based on the index.

    Parameters
    ----------
    data : list of pd.Series or pd.DataFrame
        data to be interested

    Returns
    -------
    intersection : list of pd.Series or pd.DataFrame
        intersected data

    """
    common_idx = None
    for d in data:
        if common_idx is None:
            common_idx = d.index
        else:
            common_idx = common_idx & d.index

    if len(common_idx) == 0:
        warnings.warn("Intersection is empty.")

    intersected_data = []
    for d in data:
        intersected_data.append(d.loc[common_idx])

    return intersected_data


def subsample(df, prob_drop):
    if prob_drop < 0.0:
        raise ValueError(prob_drop)
    if prob_drop > 1.0:
        raise ValueError(prob_drop)
    n = len(df)
    mask = np.random.choice([True, False], size=n, replace=True, p=[(1 - prob_drop), prob_drop])
    res = df.iloc[mask]
    return res


def check_consistent_indices(*args):
    series = list(args)
    for i in range(1, len(series)):
        if not series[0].index.equals(series[i].index):
            return False
    return True


def repeat_series(s, new_lvl_0_index, names=['date', 'asset']):
    """Repeats tha series s len(new_lvl_0_index) many times to create a new MultiIndex series

    Parameters
    ----------
    s : pd.Series
        series to be repeated. It's index label will be level 1 index label of the new series
    new_lvl_0_index :
        index level 0 of the new MultiIndex Series
    names : list[str]
         labels in the index of the new MultiIndex series

    Returns
    -------
    new MultiIndex series of size = s.size * len(new_lvl_0_index)

    """
    if isinstance(s.index, pd.MultiIndex):
        new_lvl_1_index = s.index.droplevel(0)
        assert new_lvl_1_index.name == names[1]
    else:
        new_lvl_1_index = s.index
    if not isinstance(s, pd.Series):
        raise ValueError()
    if not len(names) == 2:
        raise ValueError
    new_vals = np.tile(s.values, len(new_lvl_0_index))
    new_index = [new_lvl_0_index, new_lvl_1_index]
    new_index = pd.MultiIndex.from_product(new_index, names=names)
    return pd.Series(new_vals, index=new_index)
