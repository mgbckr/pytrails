from unittest import TestCase
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.preprocessing import normalize
from ..markovchain import MarkovChain
import h5sparse
import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))


class TestMarkovChain(TestCase):

    def test_marginal_likelihood(self):
        transition_counts = csr_matrix([[1, 2, 3], [0, 0, 0], [7, 8, 9]], dtype=np.float64)
        transition_probabilities = normalize(transition_counts, "l1", axis=1)
        pseudo_counts = transition_probabilities * 5
        ml = MarkovChain.marginal_likelihood(transition_counts, pseudo_counts, 1.0)
        print(ml)

    def test_marginal_likelihood_hdf5(self):
        """
        I don't think this gets us much because ´.value´ will probably load the complete sparse matrix
        :return:
        """
        transition_counts = csr_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        transition_probabilities = normalize(transition_counts, "l1", axis=1)
        pseudo_counts = transition_probabilities * 5

        if os.path.isfile("test.h5"):
            os.remove("test.h5")

        with h5sparse.File("test.h5") as h5f:
            h5f.create_dataset('transition/counts', data=transition_counts)
            h5f.create_dataset('pseudo/counts', data=pseudo_counts)

        h5f = h5sparse.File("test.h5")
        print(h5f['transition/counts'].h5py_group.attrs["h5sparse_shape"])
        print(h5f['transition/counts'][0:2])

        ml = MarkovChain.marginal_likelihood(
            h5f['transition/counts'].value,
            h5f['pseudo/counts'].value,
            smoothing=1.0)

        os.remove("test.h5")

        print(ml)

    def test_marginal_likelihood_inconsistent_csr_matrix(self):
        import pickle
        from pytrails.hyptrails.markovchain_variants import MarkovChain as Variants

        matrices = None
        with (open(os.path.join(WORKING_DIR, "csr_matrices_with_duplicates.pkl"), "rb")) as openfile:
            while True:
                try:
                    matrices = pickle.load(openfile, encoding="iso-8859-1")
                except EOFError:
                    break

        counts = matrices[0]
        hyp = matrices[1]

        # es = [MarkovChain.marginal_likelihood(counts, hyp * k * 1000) for k in range(9, 10)]
        # esv = [MarkovChain.marginal_likelihood_vanilla(counts, hyp * k * 1000) for k in range(9, 10)]
        # ess = [MarkovChain.marginal_likelihood_masking(counts, hyp * k * 1000) for k in range(9, 10)]

        k = 10
        reference = MarkovChain.marginal_likelihood(counts, hyp * k * 1000)
        vanilla = Variants.marginal_likelihood_vanilla(counts, hyp * k * 1000)
        masking = Variants.marginal_likelihood_masking(counts, hyp * k * 1000)

        self.assertAlmostEqual(reference, vanilla)
        self.assertAlmostEqual(reference, masking)
        self.assertAlmostEqual(vanilla, masking)



