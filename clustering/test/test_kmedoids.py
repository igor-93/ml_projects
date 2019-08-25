from collections import Counter
from unittest import TestCase, main, skip
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import scipy.stats as ss

from scipy.spatial.distance import pdist, squareform

from clustering.kmedoids import KMedoids


class TestKMedoids(TestCase):

    @skip("It has plotting, so it's not a real test...")
    def test_example(self):
        n_samples = 2000
        X, y = make_blobs(n_samples=n_samples, centers=4, cluster_std=0.3)

        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.title('Blobs')
        plt.show()

        dist_matrix = squareform(pdist(X, metric='cosine'))
        print('dist_matrix: ', dist_matrix.shape)

        plt.imshow(dist_matrix)
        plt.title('dist_matrix')
        plt.show()

        km = KMedoids(dist_matrix)

        clusters, medians = km.cluster(k=3)

        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.scatter(X[km.inited_medoids, 0], X[km.inited_medoids, 1],
                    marker='x', s=169, linewidths=3, color='k')
        plt.title('Inited medoids')
        plt.show()

        clusters = ss.rankdata(clusters, method='dense') - 1
        c = Counter(clusters)

        print('Clusters: ', c)

        plt.scatter(X[:, 0], X[:, 1], c=clusters)
        plt.scatter(X[medians, 0], X[medians, 1],
                    marker='x', s=169, linewidths=3, color='k')
        plt.title('Clusters')
        plt.show()


if __name__ == '__main__':
    main()
