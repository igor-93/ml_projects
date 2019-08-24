from collections import Counter, OrderedDict
from os.path import join, isdir
import numpy as np
import sklearn.preprocessing as skpp
from scipy.sparse import csc_matrix, isspmatrix_csr, isspmatrix_csc, save_npz
import scipy.stats as ss


def run_mcl(sim_matrix, expand_factor=2, inflate_factor=2, max_loop=40, mult_factor=1, verbose=False, save_dir=None,
            measure=None):
    """
    Main function to run Markov Clustering.
    :param sim_matrix: input square matrix that represents the graph in CSR format. The matrix is symmetric,
        and represents the similarities.
    :param expand_factor:
    :param inflate_factor:
    :param max_loop: max number of iterations
    :param mult_factor:
    :return:
    """
    if not isdir(save_dir):
        raise IOError('Directory doesnt exist: ', save_dir)

    sim_matrix = csc_matrix(sim_matrix)

    # normalize the columns so they represent the probabilities
    sim_matrix = skpp.normalize(sim_matrix, norm='l1', axis=0, copy=False)

    if not isspmatrix_csc(sim_matrix):
        raise Exception('MCL converted the sim_matrix to some other format, it is gonna be slow!!!!')

    print('MCL parameters: expand = {}, inflate = {}.'.format(expand_factor, inflate_factor))

    sim_matrix = prune(sim_matrix, verbose=verbose)
    test_counter = 0
    for i in range(max_loop):
        if verbose:
            print('Iteration %d / %d' % (i, max_loop - 1))
            print('Inflating...')
        sim_matrix = inflate(sim_matrix, inflate_factor)
        if verbose:
            print('Expanding...')
        sim_matrix = expand(sim_matrix, expand_factor)
        if verbose:
            print('Pruning...')
        sim_matrix = prune(sim_matrix, verbose=verbose)

        test_counter = i
        if stop(sim_matrix, i):
            break

    print('MCL_STOPPED at iteration %i. Getting clusters... ' % test_counter)

    sim_matrix = sim_matrix.toarray()

    labels = get_labels(sim_matrix)

    if verbose:
        print('Number of clusters: : ', len(set(labels)))
        c = Counter(labels)
        print('Clusters: ', c.most_common(20))
        # draw(sim_matrix, clusters)

    error = sum_of_squared_error(sim_matrix, is_dist=False, labels=labels)
    print('Sum of Squared Error: ', error)

    if save_dir is not None:
        filename = join(save_dir, 'mcl_{}_{}.png'.format(int(inflate_factor * 10), measure))
        save_sim_clustered(sim_matrix, labels, filename=filename)

    return labels


"""
Functions below are implementing MCL clustering.

"""


def inflate(A, inflate_factor):
    """
    Inflation step: element-wise multiply the matrix by itself.
    :param A: input matrix
    :param inflate_factor: factor by which to multiply. Usually 1.2 to 2.5
    :return: normalized matrix
    """
    if not isspmatrix_csc(A):
        raise Exception('MCL converted the sim_matrix to some other format, it is gonna be slow!!!!')
    A.data **= inflate_factor
    res = skpp.normalize(A, norm='l1', axis=0, copy=True)
    return res


def expand(A, expand_factor):
    """
    Expansion step: mulitply the matrix by itself. This is on step in a random walk.
    :param A: input matrix
    :param expand_factor: factor by which to multiply. Usually 2.
    :return: expanded matrix
    """
    if not isspmatrix_csc(A):
        raise Exception('MCL converted the sim_matrix to some other format, it is gonna be slow!!!!')
    A **= expand_factor
    return A


def regularize(M, M_g):
    """
    Used in R-MCL instead of expand.
    :param M: input matrix
    :param M_g: regularization matrix
    :return:
    """
    return np.dot(M, M_g)


def add_diag(A, mult_factor):
    """
    Adds diagonal to the matrix: A + diag(mult_factor)
    :param A: input matrix
    :param mult_factor: entries in diagonal
    :return:
    """
    return A + mult_factor * np.eye(A.shape[0])


def prune(M, verbose=False):
    """
    Pruning step: remove the very small elements to speed up the computation.
    :param M: input matrix
    :param verbose: check if it returns CSC matrix
    :return: pruned matrix
    """
    threshold_start = 1e-3
    M_temp = M.multiply(M > threshold_start)
    column_sums = M_temp.sum(axis=0)
    ave_sum_left = column_sums.sum() / float(M_temp.shape[0])
    if verbose:
        print('ave_sum_left: ', ave_sum_left)

    # we have to prune more
    threshold_lst = [5 * 1e-4, 1e-4, 5 * 1e-5, 1e-5]
    for threshold in threshold_lst:
        if ave_sum_left >= 0.8:
            break
        else:
            M_temp = M.multiply(M > threshold)
            column_sums = M_temp.sum(axis=0)
            ave_sum_left = column_sums.sum() / float(M_temp.shape[0])
            if verbose:
                print('     ave_sum_left: ', ave_sum_left)
    if ave_sum_left > 0.5:
        M = M_temp
        M = skpp.normalize(M, norm='l1', axis=0, copy=False)

    if not isspmatrix_csc(M):
        raise Exception('MCL converted the sim_matrix to some other format, it is gonna be slow!!!!')

    return M


def get_labels(sim_matrix):
    """
    Returns the labels after the algorithm has finished.
    :param sim_matrix: input matrix
    :return: labels for each common point
    """
    clusters = []
    for i, row in enumerate((sim_matrix > 0).tolist()):
        if row[i]:
            clusters.append(sim_matrix[i, :] > 0)

    clust_map = {}
    for cn, c in enumerate(clusters):
        for x in [i for i, x in enumerate(c) if x]:
            clust_map[cn] = clust_map.get(cn, []) + [x]

    labels = -1 * np.ones((sim_matrix.shape[0],), dtype=int)
    for cl_id, nodes in clust_map.items():
        for v in nodes:
            labels[v] = cl_id

    if not (labels != -1).all():
        raise Exception('Missing labels for some clusters.')

    labels = ss.rankdata(labels, method='dense') - 1

    return labels


def stop(M, i):
    """
    Check if the algorithm converged.
    :param M: input matrix
    :param i: iteration steo
    :return: boolean: True if converged
    """
    # this saves time, so we dont have to do multiplication in the first 7 iterations
    if i > 6:
        M_temp = M ** 2 - M
        m = M_temp.max() - M_temp.min()
        if abs(m) < 1e-8:
            return True

    return False


def draw(A, cluster_map={}, colors=[]):
    """
    Graph visualization with the colors of clusters.
    :param A: input matrix
    :param cluster_map: mapping of cluster used for coloring
    :param colors: if cluster_map=None, set the colors manually
    :return:
    """
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.from_numpy_matrix(A)
    if colors == []:
        if cluster_map == {}:
            print('ERROR in draw: at least of two attributes for coloring has to be provided.')
        clust_map = {}
        for k, vals in cluster_map.items():
            for v in vals:
                clust_map[v] = k

        colors = []
        for i in range(len(G.nodes())):
            colors.append(clust_map.get(i, 100))

    if len(colors) != A.shape[0]:
        print('ERROR in draw: len(colors) <> A.shape[0].')
    pos = nx.spring_layout(G)

    from matplotlib.pylab import matshow, show, cm
    plt.figure(2)
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color=colors, cmap=plt.cm.Blues)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    # matshow(A, fignum=1, cmap=cm.gray)
    plt.show()
    show()
