import numpy as np
from scipy.sparse import csr_matrix

from ..hyptrails.markovchain import MarkovChain as HypTrailsMarkovChain


class MarkovChain:

    @staticmethod
    def marginal_likelihood(
            transition_counts,
            transition_probabilities,
            concentration_factors=None,
            smoothing=1.0):
        """
        Calculates the marginal likelihood of a Markov chain based on Dirichlet priors.
        To elicit the Dirichlet prior this methods requires transition probabilities
        and an array of concentration factors.
        This method calculates a marginal likelihood for each concentration factor.

        Matrices are given as row-based Spark RDDs.

        :type transition_counts: pyspark.RDD
        :param transition_counts:

        :type transition_probabilities: RDD
        :param transition_probabilities: transition probabilities in a Spark RDD;
            the rows need to be normalized

        :type concentration_factors: np.array
        :param concentration_factors: concentration factors to elicit hypotheses;
            each concentration factor results in a separate marginal likelihood value

        :type smoothing: float
        :param smoothing: smoothing (default = 1.0)

        :return: a marginal likelihood for each given concentration factor
        """

        # align transition counts and probabilities
        aligned = transition_counts.leftOuterJoin(transition_probabilities)

        # function to calculate marginal likelihoods for each concentration factor;
        # we use row-wise elicitation
        def combine(transition_counts_row, transition_probabilities_row):
            # A None-valued entry could occur through the join.
            if transition_probabilities_row is None:
                transition_probabilities_row = csr_matrix((1, transition_counts_row.shape[1]))
            return np.array([
                HypTrailsMarkovChain.marginal_likelihood(
                    transition_counts_row,
                    transition_probabilities_row * cf,
                    smoothing)
                for cf in concentration_factors])

        # do the actual calculations
        return aligned\
            .mapValues(lambda e: combine(e[0], e[1]))\
            .values()\
            .reduce(lambda a, b: a + b)

    @staticmethod
    def marginal_likelihood_live_hypothesis(
            transition_counts_with_destination_features,
            transition_probability_function,
            concentration_factors=None,
            smoothing=1.0):

        def row_ml(row_index, row_values, destination_features):

            transition_probabilities = transition_probability_function(
                row_index,
                row_values,
                destination_features)

            return np.array([
                HypTrailsMarkovChain.marginal_likelihood(
                    row_values,
                    transition_probabilities * cf,
                    smoothing)
                for cf in concentration_factors])

        return transition_counts_with_destination_features.\
            map(lambda e: row_ml(e[0], e[1][0], e[1][1])).sum()