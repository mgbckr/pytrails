from scipy.special import gammaln
from scipy.sparse import csr_matrix
import numpy as np


class MarkovChain:

    @staticmethod
    def marginal_likelihood_vanilla(transition_counts, pseudo_counts, smoothing=1.0):

        alphas = pseudo_counts + np.array([smoothing])

        ml = 0
        for i in range(transition_counts.shape[0]):
            gamma_sum_alphas = gammaln(alphas[i].sum())
            sum_gamma_nalphas = gammaln((transition_counts[i] + alphas[i]).data).sum()
            sum_gamma_alphas = gammaln(alphas[i].data).sum()
            gamma_sum_nalphas = gammaln((transition_counts[i] + alphas[i]).A.sum())
            ml += (gamma_sum_alphas + sum_gamma_nalphas - sum_gamma_alphas - gamma_sum_nalphas)

        return ml

    @staticmethod
    def marginal_likelihood_vanilla(transition_counts, pseudo_counts, smoothing=1.0):

        if smoothing <= 0:
            raise ValueError("Smoothing must be greater than zero (Error: smoothing <= 0).")

        alphas = pseudo_counts + np.array([smoothing])

        ml = 0
        for i in range(transition_counts.shape[0]):
            gamma_sum_alphas = gammaln(alphas[i].sum())
            sum_gamma_nalphas = gammaln((transition_counts[i] + alphas[i]).data).sum()
            sum_gamma_alphas = gammaln(alphas[i].data).sum()
            gamma_sum_nalphas = gammaln((transition_counts[i] + alphas[i]).A.sum())
            ml += (gamma_sum_alphas + sum_gamma_nalphas - sum_gamma_alphas - gamma_sum_nalphas)

        return ml

    @staticmethod
    def marginal_likelihood_masking(transition_counts, pseudo_counts, smoothing=1.0):
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
        transition_pseudo_counts = transition_counts + pseudo_counts

        # marginal likelihood
        ml = 0.0
        ml += gammaln(pseudo_counts.sum(axis=1) + number_of_states * smoothing).sum()
        ml -= gammaln(transition_pseudo_counts.sum(axis=1) + number_of_states * smoothing).sum()

        transition_counts_mask = csr_matrix(
            (np.ones(transition_counts.data.shape),
             transition_counts.indices,
             transition_counts.indptr),
            transition_counts.shape)

        ml += gammaln(transition_pseudo_counts.multiply(transition_counts_mask).data + smoothing).sum()

        ml -= gammaln(pseudo_counts.multiply(transition_counts_mask).data + smoothing).sum()

        return ml
