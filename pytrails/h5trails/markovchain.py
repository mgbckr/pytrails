from ..hyptrails.markovchain import MarkovChain as HypTrailsMarkovChain
import numpy as np
from scipy.sparse import csr_matrix
import h5sparse


class MarkovChain:

    @staticmethod
    def csr_matrix_to_h5(matrix, file, name):
        """
        :type matrix: csr_matrix
        :param matrix: the matrix to parallelize

        :type file: str
        :param file: HDF5 file to hold the matrix

        :type name: str
        :param name: name of the matrix in the HDF5 file
        """
        with h5sparse.File(file) as h5f:
            h5f.create_dataset(name, data=matrix)

    @staticmethod
    def marginal_likelihood(
            transition_counts_h5file,
            transition_counts_h5name,
            transition_probabilities_h5file,
            transition_probabilities_h5name,
            concentration_factors=None,
            smoothing=1.0):
        # TODO: add chunking and parallelization?
        """

        :param transition_counts_h5file:
        :param transition_counts_h5name:
        :param transition_probabilities_h5file:
        :param transition_probabilities_h5name:
        :param concentration_factors:
        :param smoothing:
        :return:
        """

        transition_counts_h5f = h5sparse.File(transition_counts_h5file)[transition_counts_h5name]
        transition_probabilities_h5f = h5sparse.File(transition_probabilities_h5file)[transition_probabilities_h5name]

        shape = transition_counts_h5f.h5py_group.attrs["h5sparse_shape"]
        ml = np.zeros(len(concentration_factors))
        for row in range(shape[0]):
            transition_counts_row = transition_counts_h5f[row:row+1]
            transition_probabilities_row = transition_probabilities_h5f[row:row+1]
            ml_row = np.array([
                HypTrailsMarkovChain.marginal_likelihood(
                    transition_counts_row,
                    transition_probabilities_row * cf,
                    smoothing)
                for cf in concentration_factors])
            ml += ml_row

        return ml





