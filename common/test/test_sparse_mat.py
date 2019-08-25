import numpy as np
from scipy.sparse import coo_matrix

from unittest import TestCase, main

from common.sparse_mat import drop_data, drop_rows, drop_cols


class TestUtils(TestCase):

    @staticmethod
    def get_matrix():
        r1 = 4 * [1] + 6 * [0]
        r2 = [0] + 4 * [1] + 5 * [0]
        r3 = 2 * [1] + 4 * [0] + 2 * [1] + 2 * [0]
        r4 = 5 * [1] + 5 * [0]
        r5 = 6 * [0] + 3 * [1] + [0]
        mat = [r1, r2, r3, r4, r5]
        mat = np.asarray(mat, dtype=int)
        # print('Test matrix:')
        # print(mat)
        return mat

    def test_drop_data(self):
        mat = self.get_matrix()
        mat = np.multiply(mat, np.random.randint(10, size=mat.shape))
        threshold = 5
        mat = coo_matrix(mat)
        res = drop_data(mat, threshold)
        passed = (res.data >= threshold).all()
        passed2 = (mat.data >= threshold).all()
        self.assertTrue(passed)
        self.assertTrue(passed2)

    def test_drop_cols(self):
        mat = self.get_matrix()
        M = coo_matrix(mat)
        idx_to_drop = [7, 4, 2]

        res = drop_cols(M, idx_to_drop)

        r1 = 3 * [1] + 4 * [0]
        r2 = [0] + 2 * [1] + 4 * [0]
        r3 = 2 * [1] + 2 * [0] + 1 * [1] + 2 * [0]
        r4 = 3 * [1] + 4 * [0]
        r5 = 4 * [0] + 2 * [1] + [0]
        expected = [r1, r2, r3, r4, r5]
        expected = np.asarray(expected, dtype=int)
        passed = (expected.shape == res.shape)
        res = res.todense()
        passed = passed and np.array_equal(res, expected)
        self.assertTrue(passed)

    def test_drop_rows(self):
        mat = self.get_matrix()
        M = coo_matrix(mat)
        idx_to_drop = [1, 4, 3]

        res = drop_rows(M, idx_to_drop)

        r1 = 4 * [1] + 6 * [0]
        r3 = 2 * [1] + 4 * [0] + 2 * [1] + 2 * [0]
        expected = [r1, r3]
        expected = np.asarray(expected, dtype=int)
        passed = expected.shape == res.shape
        res = res.todense()
        passed = passed and np.array_equal(res, expected)
        self.assertTrue(passed)


if __name__ == '__main__':
    main()
