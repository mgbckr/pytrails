from scipy.special import gammaln
from scipy.sparse import csr_matrix


class MarkovChain:

    @staticmethod
    def marginal_likelihood(transition_counts, pseudo_counts, smoothing=1.0):
        """
        Calculates the marginal likelihood for a first-order Markov chain based on Dirichlet priors.
        The Dirichlet priors are given as a pseudo count matrix.

        Matrices are given as ´scipy.sparse.csr_matrix´.

        Inspired by https://github.com/sequenceanalysis/sequenceanalysis.github.io/blob/master/notebooks/part4.ipynb

        :type transition_counts: csr_matrix
        :param transition_counts: transition counts between states

        :type pseudo_counts: csr_matrix
        :param pseudo_counts: elicited hypothesis in the form of pseudo counts WITHOUT smoothing

        :type smoothing: float
        :param smoothing: smoothing (default = 1.0)

        :return marginal likelihood of a first-order Markov chain based on Dirichlet priors
        """

        number_of_states = transition_counts.shape[1]

        transitions_prior = transition_counts + pseudo_counts

        # marginal likelihood
        ml = 0.0
        ml += gammaln(pseudo_counts.sum(axis=1) + number_of_states * smoothing).sum()
        ml -= gammaln(transition_counts.sum(axis=1) + pseudo_counts.sum(axis=1) + number_of_states * smoothing).sum()

        # add summed up smoothed log gamma entries where either the transition count or the hypothesis is != 0
        ml += gammaln(transitions_prior.data + smoothing).sum()

        # calculate the sum of log gamma entries for the hypothesis
        # where the transition count > 0 and alpha == 0 (without smoothing)
        gamma_smoothing = 0.0
        if smoothing > 0:
            gamma_smoothing = (len(transitions_prior.data) - len(pseudo_counts.data)) * gammaln(smoothing)

        # subtract summed up log gamma entries of hypothesis,
        # which is the sum of the hypothesis entries and the entries
        # where the transition count > 0 and alpha == 0 (without smoothing)
        ml -= gammaln(pseudo_counts.data + smoothing).sum() + gamma_smoothing

        return ml
