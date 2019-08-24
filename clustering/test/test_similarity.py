import numpy as np
from scipy.sparse import csr_matrix
from unittest import TestCase, main

from ..similarity import get_similarity, SimMeasure


class TestSimilarity(TestCase):

    @staticmethod
    def get_matrix():
        r1 = 4 * [1] + 6 * [0]
        r2 = [0] + 4 * [1] + 5 * [0]
        r3 = 2 * [1] + 4 * [0] + 2 * [1] + 2 * [0]
        r4 = 5 * [1] + 5 * [0]
        r5 = 6 * [0] + 3 * [1] + [0]
        mat = [r1, r2, r3, r4, r5]
        mat = np.asarray(mat, dtype=int)
        # print(mat)
        return mat

    def test_cosine(self):
        mat = self.get_matrix()
        cosine = [[0, 3 / 4, 0.5, 4 / (2 * np.sqrt(5)), 0],
                  [0, 0, 0.25, 4 / (2 * np.sqrt(5)), 0],
                  [0, 0, 0, 2 / (2 * np.sqrt(5)), 2 / (2 * np.sqrt(3))],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]
        cosine = np.asarray(cosine, dtype=float)
        cosine = (cosine.T + cosine) * 0.9999 + np.eye(5, dtype=float)
        cosine = csr_matrix(cosine)
        res = get_similarity(mat, SimMeasure.COSINE)
        self.assertEqual((res != cosine).nnz, 0)

    def test_jaccard(self):
        mat = self.get_matrix()
        jaccard = [[0, 3 / 5, 2 / 6, 4 / 5, 0],
                   [0, 0, 1 / 7, 4 / 5, 0],
                   [0, 0, 0, 2 / 7, 2 / 5],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        jaccard = np.asarray(jaccard, dtype=float)
        jaccard = (jaccard.T + jaccard) * 0.9999 + np.eye(5, dtype=float)
        jaccard = csr_matrix(jaccard)
        res = get_similarity(mat, SimMeasure.JACCARD)
        self.assertEqual((res != jaccard).nnz, 0)

    def test_jaccard3(self):
        mat = self.get_matrix()
        jaccard3 = [[0, 9 / 11, 6 / 10, 12 / 13, 0],
                    [0, 0, 1 / 3, 12 / 13, 0],
                    [0, 0, 0, 6 / 11, 2 / 3],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]]
        jaccard3 = np.asarray(jaccard3, dtype=float)
        jaccard3 = (jaccard3.T + jaccard3) * 0.9999 + np.eye(5, dtype=float)
        jaccard3 = csr_matrix(jaccard3)

        res = get_similarity(mat, SimMeasure.JACCARD3)
        self.assertEqual((res != jaccard3).nnz, 0)

    def test_johnson(self):
        mat = self.get_matrix()
        johnson = [[0, 0.75, 0.5, 0.9, 0],
                   [0, 0, 0.25, 0.9, 0],
                   [0, 0, 0, 0.45, 7 / 12],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        johnson = np.asarray(johnson, dtype=float)
        johnson = (johnson.T + johnson) * 0.9999 + np.eye(5, dtype=float)
        johnson = csr_matrix(johnson)
        res = get_similarity(mat, SimMeasure.JOHNSON)
        self.assertEqual((res != johnson).nnz, 0)

    def test_simpson(self):
        mat = self.get_matrix()
        simpson = [[0, 0.75, 0.5, 1, 0],
                   [0, 0, 0.25, 1, 0],
                   [0, 0, 0, 0.5, 2 / 3],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        simpson = np.asarray(simpson, dtype=float)
        simpson = (simpson.T + simpson) * 0.9999 + np.eye(5, dtype=float)
        simpson = csr_matrix(simpson)
        res = get_similarity(mat, SimMeasure.SIMPSON)
        self.assertEqual((res != simpson).nnz, 0)


if __name__ == '__main__':
    main()
