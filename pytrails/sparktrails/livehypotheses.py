import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


def uniform_hypothesis(row_index, row_values, destination_features):
    return csr_matrix(normalize(np.ones(row_values.shape[1]).reshape(1, -1), "l1", axis=1))


def uniform_no_selftransitions_hypothesis(row_index, row_values, destination_features):
    ones = np.ones(row_values.shape[1])
    ones[row_index] = 0
    return csr_matrix(normalize(ones.reshape(1, -1), "l1", axis=1))
