from unittest import TestCase, main
import numpy as np
import pandas as pd
from numba import jit

from common.pandas_utils import column_name_to_id, rolling_multi_columns

TOL = 1e-9


class ColumnsTest(TestCase):

    def test_simple(self):
        df_cols = ['col1 O', 'col1 H', 'col1 L', 'col1 C',
                   'col2 O', 'col2 H', 'col2 L', 'col2 C',
                   'col3 O', 'col3 H', 'col3 L', 'col3 C',]

        open_cols = ['col1 O', 'col2 O', 'col3 O']
        close_cols = ['col1 C', 'col2 C', 'col3 C']
        high_cols = ['col1 H', 'col2 H', 'col3 H']

        open_ids, high_ids, close_ids = column_name_to_id(df_cols, open_cols, high_cols, close_cols)
        self.assertEqual([0, 4, 8], open_ids)
        self.assertEqual([1, 5, 9], high_ids)
        self.assertEqual([3, 7, 11], close_ids)


class RollingMultiColumnsTest(TestCase):

    def test1(self):
        @jit(nopython=True)
        def fn(x):
            s = x.sum(axis=0)
            return s.sum()
        df = pd.DataFrame(index=[0,1,2])
        df["a"] = [1, 2, 3]
        df["b"] = [10, 20, 30]
        df["c"] = [100, 200, 300]
        print(df)

        out = rolling_multi_columns(df, ["a", "b"], window=2, fn=fn)
        expected = pd.Series([np.nan, 33, 55], index=[0, 1, 2])

        equal = out.equals(expected)
        if not equal:
            print(out)
            print(expected)

        self.assertTrue(equal)


if __name__ == '__main__':
    main()
