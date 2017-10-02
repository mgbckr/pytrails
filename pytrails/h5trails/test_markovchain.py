from unittest import TestCase
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from . import MarkovChain
import os


class TestH5MarkovChain(TestCase):

    def test_marginal_likelihood(self):

        file = "test.h5"
        if os.path.isfile(file):
            os.remove(file)

        transition_counts = csr_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        transition_probabilities = normalize(transition_counts, "l1", axis=1)

        MarkovChain.csr_matrix_to_h5(transition_counts, file, "tc")
        MarkovChain.csr_matrix_to_h5(transition_probabilities, file, "tp")

        ml = MarkovChain.marginal_likelihood(file, "tc", file, "tp", [0, 1, 2, 5], 1.0)

        print(ml)

        os.remove(file)
