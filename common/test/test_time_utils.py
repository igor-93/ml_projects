from unittest import TestCase, main
import numpy as np
import pandas as pd

from common.time_utils import normalize_tz, infer_freq


class TestInferFreq(TestCase):

    def test_minute_full(self):
        idx = pd.date_range('2019-1-1 12:00:00', '2019-1-1 14:00:00', freq='1min')
        freq, is_missing = infer_freq(idx, return_if_missing=True)
        self.assertEqual(freq, 'T')
        self.assertFalse(is_missing)

    def test_minute_missing(self):
        idx = ['2019-1-1 12:00:00', '2019-1-1 12:01:00', '2019-1-1 12:02:00',
               '2019-1-1 12:03:00', '2019-1-1 12:05:00', '2019-1-1 12:06:00']
        idx = pd.DatetimeIndex(idx)
        freq, is_missing = infer_freq(idx, return_if_missing=True)
        self.assertEqual(freq, 'T')
        self.assertTrue(is_missing)

    def test_sec_full(self):
        idx = pd.date_range('2019-1-1 12:00:00', '2019-1-1 14:00:00', freq='1s')
        freq, is_missing = infer_freq(idx, return_if_missing=True)
        self.assertEqual(freq, 'S')
        self.assertFalse(is_missing)

    def test_sec_missing(self):
        idx = ['2019-1-1 12:00:00', '2019-1-1 12:00:01', '2019-1-1 12:00:02',
               '2019-1-1 12:00:03', '2019-1-1 12:00:05', '2019-1-1 12:00:06']
        idx = pd.DatetimeIndex(idx)
        freq, is_missing = infer_freq(idx, return_if_missing=True)
        self.assertEqual(freq, 'S')
        self.assertTrue(is_missing)


class TestDST(TestCase):

    def get_data(self):
        idx1 = pd.date_range('2018-9-1 13:30', '2018-9-1 20:00', freq='10min')
        idx1_norm = pd.date_range('2018-9-1 1430', '2018-9-1 21:00', freq='10min')
        n1 = len(idx1)
        df1 = pd.DataFrame(np.arange(n1), index=idx1, columns=['common'])
        df1['GMT Offset'] = -4
        df1_norm = pd.DataFrame(np.arange(n1), index=idx1_norm, columns=['common'])
        df1_norm['GMT Offset'] = -5

        idx2 = pd.date_range('2018-11-4 14:30', '2018-11-4 21:00', freq='10min')
        n2 = len(idx2)
        df2 = pd.DataFrame(np.arange(n2), index=idx2, columns=['common'])
        df2['GMT Offset'] = -5

        df = pd.concat([df1, df2], axis=0, sort=True)
        df_norm = pd.concat([df1_norm, df2], axis=0, sort=True)
        return df, df_norm

    def test_wo_gmt_offset(self):
        df, df_norm = self.get_data()
        df.drop(columns=['GMT Offset'], inplace=True)
        df_norm.drop(columns=['GMT Offset'], inplace=True)
        out_df = normalize_tz(df=df, timezone='US/Eastern')

        self.assertEqual(out_df.shape, df.shape)

        is_eq = out_df.equals(df_norm)
        self.assertTrue(is_eq)


if __name__ == '__main__':
    main()
