import pandas as pd
import numpy as np

from common.array_utils import split_array_3


class TimeSeriesCV:
    """
    Time-series Cross Validation. Similar to scikit-learn CV, but it implements purging.
    """

    def __init__(self, train_size, validation_size, test_size, label_dt, max_k=None,
                 allow_larger_train_size=False, loc_or_iloc="loc"):
        """
        Sliding window cross-validation. Train and test sets are sliding in each iteration and validation set is fixed
        to be after the last slide of test set.  NOTE: for larger time periods use np.timedelta64. E.g. 36 months is
        np.timedelta64(36, 'M').

        Parameters
        ----------
        train_size : pd.Timedelta, timedelta, np.timedelta64 or string.
            size of the training set.
        validation_size : pd.Timedelta, timedelta, np.timedelta64 or string.
            size of the test set.
        test_size : pd.Timedelta, timedelta, np.timedelta64 or string.
            size of the validation set. This set is excluded from the split().
        label_dt : pd.Timedelta, timedelta, np.timedelta64 or string.
            time period that label spans into the future. For example '1d' if we predict up to 1 day price change.
            NOTE: this must be given in real time, no matter if input time series is only trading minutes or not.
        max_k : int
            maximum number of folds. Optional. If given, only most recent max_k folds will be created.
        allow_larger_train_size : bool, optional
            if True, train_size means the minimal allowed train size, and it increases with each fold.
            If False, train_size will remain constant across the folds
        loc_or_iloc : str, "loc" or "iloc"
            if loc, the split rerturns indices of the same type as df.index. If iloc it returns
            integer indices (.iloc should be used to acces elements of train/test set)

        """
        self.train_size = pd.Timedelta(train_size)
        self.validation_size_ = pd.Timedelta(validation_size)
        self.test_size_ = pd.Timedelta(test_size)
        self.label_dt = pd.Timedelta(label_dt)

        if self.train_size is pd.NaT:
            raise ValueError('train_size is wrong: ', train_size)
        if self.validation_size_ is pd.NaT:
            raise ValueError('validation_size is wrong: ', validation_size)
        if self.test_size_ is pd.NaT:
            raise ValueError('test_size is wrong: ', test_size)
        if self.label_dt is pd.NaT:
            raise ValueError('label_dt is wrong: ', label_dt)

        if max_k is not None and max_k < 1:
            raise ValueError('Invalid value for max_k: ', max_k)
        else:
            self.max_k = max_k

        self.k = None
        self.last_timestamp = None
        self.last_test_t = None
        self.allow_larger_train_size = allow_larger_train_size

        if loc_or_iloc not in ["loc", "iloc"]:
            raise ValueError(f"loc_or_iloc is wrong: {loc_or_iloc}")
        else:
            self.loc_or_iloc = loc_or_iloc

    def split(self, *dfs):
        """
        Splits common in train, test sets

        Parameters
        ----------
        dfs : pd.DataFrame
            DataFrames to the split on

        Returns
        -------
        list
            list of tuples of pd.IndexSlice: (train_slice, test_slice)

        """
        if isinstance(dfs[0], list):
            dfs = dfs[0]

        if not isinstance(dfs[0], pd.DataFrame) and not isinstance(dfs[0], pd.Series):
            raise TypeError('dfs[0] is of type ', type(dfs[0]))

        if self.loc_or_iloc == "iloc":
            idx_fst = dfs[0].index
            for df in dfs:
                if not idx_fst.equals(df.index):
                    print("# dfs: ", len(dfs))
                    print(idx_fst)
                    print(df.index)
                    raise ValueError(f"all indices must be the same if loc_or_iloc={self.loc_or_iloc}")

            time_to_id = pd.Series(np.arange(len(dfs[0])), index=idx_fst)
        else:
            time_to_id = None

        fst_timestamp = dfs[0].index[0]
        last_timestamp = dfs[0].index[-1]
        if not fst_timestamp < last_timestamp:
            raise ValueError('Index is sorted in the opposite direction or it is just wrong...', dfs[0].index)

        for i in range(1, len(dfs)):
            fst_timestamp = min(fst_timestamp, dfs[i].index[0])
            last_timestamp = max(last_timestamp, dfs[i].index[-1])

        print('First date found in common: ', fst_timestamp)
        print('Last date found in common: ', last_timestamp)

        if fst_timestamp >= last_timestamp:
            raise AssertionError('fst_timestamp >= last_timestamp ', fst_timestamp, last_timestamp)

        fst_test_t = fst_timestamp + self.train_size + self.label_dt

        last_test_t = last_timestamp - self.test_size_ - self.label_dt
        val_label_size = self.validation_size_ + self.label_dt
        k = (last_test_t + self.label_dt - fst_test_t) // val_label_size   # if last test set is shorter, throw it away

        if k < 1:
            print('k: {}, fst_timestamp: {}, last_timestamp: {}, fst_test_t: {}'.format(
                k, fst_timestamp, last_timestamp, fst_test_t))
            raise ValueError('CV needs more common for train_size: {}, test_size: {}, validation_size: {}'.format(
                self.train_size, self.validation_size_, self.test_size_))

        self.fst_timestamp = fst_timestamp
        self.last_timestamp = last_timestamp
        self.last_test_t = last_test_t

        if self.max_k is not None and self.max_k < k:
            k = self.max_k

        self.k = k

        result = []

        for i in range(k):
            test_end = last_test_t
            test_start = test_end - self.validation_size_
            test_slice = pd.IndexSlice
            test_slice = test_slice[test_start:test_end]

            train_end = test_start - self.label_dt
            train_start = train_end - self.train_size
            last_test_t = test_start - self.label_dt

            train_slice = pd.IndexSlice
            if self.allow_larger_train_size:
                train_slice = train_slice[:train_end]
            else:
                train_slice = train_slice[train_start:train_end]

            if time_to_id is not None:
                train_slice = time_to_id.loc[train_slice].values
                test_slice = time_to_id.loc[test_slice].values

            result.append((train_slice, test_slice))

        result = result[::-1]

        return result

    def get_train_and_test(self):
        """
        After splitting that should be used for Grid Search, this function returns the last part of the common: validation
        set.

        Returns
        -------
        tuple of pd.IndexSlice
            (train_slice, test_slice)

        """

        if self.last_test_t is None:
            raise AssertionError('First call split() to generate train/test splits, then get_validation().')
        validation_start = self.last_test_t + self.label_dt
        validation_end = self.last_timestamp
        train_end = self.last_test_t
        train_start = train_end - self.train_size

        train_slice = pd.IndexSlice
        train_slice = train_slice[train_start:train_end]

        val_slice = pd.IndexSlice
        val_slice = val_slice[validation_start:validation_end]

        return train_slice, val_slice


def train_test_split_random(all_indices, test_size=0.25, batch_perc=0.01, label_size=24, seed=None):
    """Randomly splits indices into train and test data. Test data consist of multiple smaller batches of size
    batch_perc. The split is purged by giving label_size > 0.

    Parameters
    ----------
    all_indices : array-like of int
        list of indices
    test_size : int or float
        size of the test set. Either given as number of data points of as a fraction of whole data set.
    batch_perc : int for float
        size of small batches that together give test set. Either given as number of data points of as a fraction of
        whole data set.
    label_size : int
        number of data points that will be removed from the beginning of each test batch and each train batch that
        comes after a test batch
    seed : int
        random seed

    Returns
    -------
    train_indices : array-like
        indices of train set
    test_indices : array-like
        indices of test set
    """
    if 0 < test_size < 1:
        pass
    elif test_size > 1 and type(test_size) == int:
        test_size = test_size / len(all_indices)
    else:
        raise ValueError(f"test_size: {test_size}")

    # label_size is num common points to be removed from every batch to avoid label overlap
    all_indices = all_indices.astype(dtype=int)
    np.random.seed(seed=seed)
    n = all_indices.shape[0]
    n_batches = int(test_size / batch_perc)
    batch_size = int(n * batch_perc)
    print(f"batch_size: {batch_size}")
    if label_size > 0.5 * batch_size:
        raise ValueError("batch_size is too small: {}".format(batch_size))
    train_indices_sets = [np.arange(n, dtype=int)]
    test_indices = np.array([], dtype=int)
    for i_batch in range(n_batches):
        splittable_indices = [i for i, s in enumerate(train_indices_sets) if
                              len(s) > 2 * batch_size]
        chosen_subset_id = np.random.choice(splittable_indices)
        chosen_subset_to_split = train_indices_sets[chosen_subset_id]
        # TODO: remove all the asserts and write the tests
        assert np.array_equal(chosen_subset_to_split, np.unique(chosen_subset_to_split))
        assert (np.diff(chosen_subset_to_split) == 1).all()
        if np.in1d(chosen_subset_to_split, test_indices).any():
            print(f"chosen_subset_to_split: {chosen_subset_to_split}")
            print(f"test_indices: {test_indices}")
            raise ValueError("Chosen subset to split already in test!!!")

        left_rest, chosen_test_batch, right_rest = split_array_3(chosen_subset_to_split, batch_size)

        assert len(chosen_test_batch) == batch_size
        assert np.in1d(chosen_test_batch, chosen_subset_to_split).all()

        if np.in1d(chosen_test_batch, test_indices).any():
            print(f"test_indices: {test_indices}")
            print(f"chosen_test_batch: {chosen_test_batch}")
            raise AssertionError

        assert len(np.intersect1d(left_rest, test_indices)) == 0
        assert len(np.intersect1d(right_rest, test_indices)) == 0
        assert len(np.intersect1d(left_rest, right_rest)) == 0
        assert len(np.intersect1d(left_rest, chosen_test_batch)) == 0
        assert len(np.intersect1d(right_rest, chosen_test_batch)) == 0

        train_indices_sets[chosen_subset_id] = left_rest  # put the left rest on the place of the old one
        reduced_right_rest = right_rest[label_size:]    # purge the train set as well
        train_indices_sets.append(reduced_right_rest)
        reduced_test_batch = chosen_test_batch[label_size:]
        test_indices = np.append(test_indices, reduced_test_batch)

    train_indices = np.concatenate(train_indices_sets)
    train_indices.sort()
    assert np.array_equal(train_indices, np.unique(train_indices))

    test_indices.sort()
    assert np.array_equal(test_indices, np.unique(test_indices))

    assert len(np.intersect1d(train_indices, test_indices, assume_unique=True)) == 0
    return train_indices, test_indices


if __name__ == '__main__':
    train_idx, test_idx = train_test_split_random(np.arange(2500), label_size=10)
