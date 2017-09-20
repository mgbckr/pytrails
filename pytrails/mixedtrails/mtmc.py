from scipy.sparse import csr_matrix

class MTMC:

    @staticmethod
    def marginal_likelihood(
            transitions,
            group_assignment_probabilities,
            transition_probabilities,
            concentration_factors,
            smoothing=1.0):
        """
        Calculates the marginal likelihood of the Mixed Transition Markov Chain (MTMC) model.

        :type group_assignment_probabilities: csr_matrix
        :param group_assignment_probabilities:

        :type transition_probabilities: np.array[csr_matrix]
        :param transition_probabilities:

        :type concentration_factors: np.array
        :param concentration_factors:

        :type smoothing: float
        :param smoothing:

        :return:
        """
        return None
