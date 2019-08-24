import numpy as np
from unittest import TestCase, main

from ..array_utils import find_runs, split_array_3


class TestFundRuns(TestCase):

    def test1(self):
        a = [1, 1, 1, 2, 3, 3, 4, 4, 4]
        exp_values = np.array([1, 2, 3, 4])
        exp_starts = np.array([0, 3, 4, 6])
        exp_lens = np.array([3, 1, 2, 3])

        out = find_runs(a)
        self.assertTrue(np.array_equal(exp_values, out[0]))
        self.assertTrue(np.array_equal(exp_starts, out[1]))
        self.assertTrue(np.array_equal(exp_lens, out[2]))


class TestRandomSplit(TestCase):

    def test_split_in_3(self):
        input = np.random.random(31)
        input = np.unique(input)

        mid_size = 6
        l, m, r = split_array_3(input, mid_size)
        self.assertEqual(len(m), mid_size)
        self.assertEqual(len(l) + len(m) + len(r), len(input))
        self.assertTrue(len(np.intersect1d(l, m)) == 0)
        self.assertTrue(len(np.intersect1d(m, r)) == 0)
        self.assertTrue(len(np.intersect1d(l, r)) == 0)

        mid_size = 29
        l, m, r = split_array_3(input, mid_size)
        self.assertEqual(len(m), mid_size)
        self.assertEqual(len(l) + len(m) + len(r), len(input))
        self.assertTrue(len(np.intersect1d(l, m)) == 0)
        self.assertTrue(len(np.intersect1d(m, r)) == 0)
        self.assertTrue(len(np.intersect1d(l, r)) == 0)

    def test_split_in_3_too_big_mid(self):
        input = np.random.random(31)
        input = np.unique(input)

        with self.assertRaises(ValueError):
            mid_size = 30
            l, m, r = split_array_3(input, mid_size)


if __name__ == '__main__':
    main()
