import gc
import itertools
import numpy as np


class KMedoids:

    def __init__(self, dist_matrix, init_type='k++'):
        self.dist_matrix = np.asarray(dist_matrix)
        self.n_points = dist_matrix.shape[0]

        self.inited_medoids = None
        if init_type not in ['k++', 'random']:
            raise ValueError('init_type argument must be either k++ or random, but it is ', init_type)
        self.init_type = init_type

    def cluster(self, k):
        """
        Performs actual clustering.
        :param k: number of clusters
        :return: ids of clusters and ids of medoids
        """
        print('Running k-Medoids with k = {}, init = {}'.format(k, self.init_type))

        if self.init_type == 'random':
            curr_medoids = self.__init_random(k)
        elif self.init_type == 'k++':
            curr_medoids = self.__init_kplusplus(k)
        else:
            raise AssertionError(self.init_type)

        # used to plot later
        self.inited_medoids = curr_medoids
        # Doesn't matter what we initialize these to.
        old_medoids = np.array([-1] * k)
        new_medoids = np.array([-1] * k)
        print('Medoids initialized: ', curr_medoids)

        it = 0
        # Until the medoids stop updating, do the following:
        while not ((old_medoids == curr_medoids).all()):
            print('Running iter %d' % it)
            # Assign each point to cluster with closest medoid.
            clusters = self.__assign_points_to_clusters(curr_medoids)

            # Update cluster medoids to be lowest cost point.
            for curr_medoid in curr_medoids:
                cluster = np.where(clusters == curr_medoid)[0]
                new_medoids[curr_medoids == curr_medoid] = self.__compute_new_medoid(cluster)

            old_medoids[:] = curr_medoids[:]
            curr_medoids[:] = new_medoids[:]

            it += 1

        gc.collect()
        return clusters, curr_medoids

    def __assign_points_to_clusters(self, medoids):
        """
        Assign each point to cluster with closest medoid.
        :param medoids: IDs of medoids
        :param distances: distance matrix
        :return:
        """
        distances_to_medoids = self.dist_matrix[:, medoids]

        clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
        clusters[medoids] = medoids
        return clusters

    def __compute_new_medoid(self, cluster):
        """
        Computes the new medoid for the given cluster
        :param cluster:
        :param distances:
        :return:
        """
        mask = np.ones(self.dist_matrix.shape)
        mask[np.ix_(cluster, cluster)] = 0.
        cluster_distances = np.ma.masked_array(data=self.dist_matrix, mask=mask, fill_value=10e9)
        costs = cluster_distances.sum(axis=1)
        return costs.argmin(axis=0, fill_value=10e9)

    def __init_random(self, k):
        # Pick k random, unique medoids.
        curr_medoids = np.array([-1] * k)
        while not len(np.unique(curr_medoids)) == k:
            curr_medoids = np.random.randint(0, self.n_points - 1, k)

        return curr_medoids

    def __init_kplusplus(self, k):
        """
        Initialize the medoids with kmeans++ algorithm by David Arthur and Sergei Vassilvitskii. This is preferd
        initialization over pure random.
        :param k: number of clusters
        :return: ids of data points that are going to be used as medoids
        """

        medoids = np.empty((k,), dtype=int)

        fst_mean = np.random.randint(0, self.n_points)
        medoids[0] = fst_mean

        # get square distances of all points to the mean
        # dists = dist(common, means[0, np.newaxis], xtx)
        dists = self.__dist_id(np.arange(self.n_points), fst_mean)

        probs = np.empty(self.n_points)

        for i in range(1, k):
            # sample a new mean weighted by squared dists
            np.divide(dists, np.linalg.norm(dists, ord=1), out=probs)
            new_mean_idx = np.random.choice(self.n_points, replace=False, p=probs)

            # add new mean
            medoids[i] = new_mean_idx

            # calculate new distances to the closest means
            new_dists = self.__dist_id(np.arange(self.n_points), new_mean_idx)

            dists = np.minimum(dists, new_dists)

        return medoids

    def __dist_id(self, list1, list2):
        """
        Returns pair-wise distances between data points with ids in list1 and ids in list2.
        :param list1: ids of data points in the first set
        :param list2: ids of data points in the second set
        :return: result[i, j] = distance between data[list1[i]] and data[list2[j]]
        """

        if not np.isscalar(list2):
            c = list(itertools.product(list1, list2))
            c1, c2 = zip(*c)
            res = np.asarray(self.dist_matrix[c1, c2]).reshape((len(list1), len(list2)))
        else:
            c1 = list1
            c2 = list2
            res = np.asarray(self.dist_matrix[c1, c2]).squeeze()

        return res
