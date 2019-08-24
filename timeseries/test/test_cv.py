import sys
import numpy as np
import pandas as pd
from unittest import TestCase, main

from timeseries.cv import TimeSeriesCV


class TestCV(TestCase):

    def test_all_k(self):
        idx = pd.date_range('2018-1-1 01:01', '2018-1-1 21:01', freq='1min')
        n = len(idx)
        df1 = pd.DataFrame(np.arange(1, n+1), index=idx, columns=['df1'])
        df2 = pd.DataFrame(np.arange(1, n+1)*10, index=idx, columns=['df2'])

        label_dt = '10min'
        train_size = '100min'
        test_size = '20min'
        validation_size = '60min'
        cv = TimeSeriesCV(train_size=train_size, validation_size=test_size, test_size=validation_size, label_dt=label_dt)

        splits = cv.split(df1, df2)
        self.assertEqual(len(splits), 34)

        self.assertTrue(splits[0][0].start >= idx[0])
        last_test_e = None
        for train_slice, test_slice in splits:
            train_s, train_e, test_s, test_e = train_slice.start, train_slice.stop, test_slice.start, test_slice.stop
            #print('train ', train_s, train_e)
            #print('test ', test_s, test_e)
            self.assertEqual(train_e, train_s + pd.Timedelta(train_size))
            self.assertEqual(test_s, train_e + pd.Timedelta(label_dt))
            self.assertEqual(test_e, test_s + pd.Timedelta(test_size))
            last_test_e = test_e

        tr_slice, val_slice = cv.get_train_and_test()
        tr_s, tr_e, val_s, val_e = tr_slice.start, tr_slice.stop, val_slice.start, val_slice.stop
        self.assertEqual(last_test_e + pd.Timedelta(label_dt), val_s)
        self.assertEqual(tr_e, tr_s + pd.Timedelta(train_size))
        self.assertEqual(val_s, tr_e + pd.Timedelta(label_dt))
        self.assertEqual(val_e, idx[-1])
        self.assertEqual(val_s, idx[-1] - pd.Timedelta(validation_size))

    def test_max_k(self):
        idx = pd.date_range('2018-1-1 01:01', '2018-1-1 21:01', freq='1min')
        n = len(idx)
        df1 = pd.DataFrame(np.arange(1, n+1), index=idx, columns=['df1'])
        df2 = pd.DataFrame(np.arange(1, n+1)*10, index=idx, columns=['df2'])

        max_k = 2
        label_dt = '10min'
        train_size = '100min'
        test_size = '20min'
        validation_size = '60min'
        cv = TimeSeriesCV(train_size=train_size, validation_size=test_size, test_size=validation_size, label_dt=label_dt, max_k=max_k)

        splits = cv.split(df1, df2)
        self.assertEqual(len(splits), max_k)
        last_test_e = None
        for train_slice, test_slice in splits:
            train_s, train_e, test_s, test_e = train_slice.start, train_slice.stop, test_slice.start, test_slice.stop
            self.assertEqual(train_e, train_s + pd.Timedelta(train_size))
            self.assertEqual(test_s, train_e + pd.Timedelta(label_dt))
            self.assertEqual(test_e, test_s + pd.Timedelta(test_size))
            last_test_e = test_e

        tr_slice, val_slice = cv.get_train_and_test()
        tr_s, tr_e, val_s, val_e = tr_slice.start, tr_slice.stop, val_slice.start, val_slice.stop
        self.assertEqual(last_test_e + pd.Timedelta(label_dt), val_s)
        self.assertEqual(tr_e, tr_s + pd.Timedelta(train_size))
        self.assertEqual(val_s, tr_e + pd.Timedelta(label_dt))
        self.assertEqual(val_e, idx[-1])
        self.assertEqual(val_s, idx[-1] - pd.Timedelta(validation_size))

    def test_different_indices(self):
        idx1 = pd.date_range('2018-1-1 01:01', '2018-1-1 19:01', freq='1min')
        idx2 = pd.date_range('2018-1-1 12:01', '2018-1-1 21:01', freq='1min')
        n1 = len(idx1)
        n2 = len(idx2)
        df1 = pd.DataFrame(np.arange(1, n1 + 1), index=idx1, columns=['df1'])
        df2 = pd.DataFrame(np.arange(1, n2 + 1) * 10, index=idx2, columns=['df2'])

        label_dt = '10min'
        train_size = '100min'
        test_size = '20min'
        validation_size = '60min'
        cv = TimeSeriesCV(train_size=train_size, validation_size=test_size, test_size=validation_size, label_dt=label_dt, max_k=5)

        splits = cv.split(df1, df2)
        last_test_e = None
        for train_slice, test_slice in splits:
            train_s, train_e, test_s, test_e = train_slice.start, train_slice.stop, test_slice.start, test_slice.stop
            self.assertEqual(train_e, train_s + pd.Timedelta(train_size))
            self.assertEqual(test_s, train_e + pd.Timedelta(label_dt))
            self.assertEqual(test_e, test_s + pd.Timedelta('20min'))
            last_test_e = test_e

        tr_slice, val_slice = cv.get_train_and_test()
        tr_s, tr_e, val_s, val_e = tr_slice.start, tr_slice.stop, val_slice.start, val_slice.stop
        self.assertEqual(last_test_e + pd.Timedelta(label_dt), val_s)
        self.assertEqual(tr_e, tr_s + pd.Timedelta(train_size))
        self.assertEqual(val_s, tr_e + pd.Timedelta(label_dt))
        self.assertEqual(val_e, idx2[-1])
        self.assertEqual(val_s, idx2[-1] - pd.Timedelta(validation_size))


if __name__ == '__main__':
    main()
