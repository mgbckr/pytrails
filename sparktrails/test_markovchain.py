from unittest import TestCase
import pyspark
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sparktrails.markovchain import MarkovChain


class TestSparkMarkovChain(TestCase):

    def test_marginal_likelihood(self):

        transition_counts = csr_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        transition_probabilities = normalize(transition_counts, "l1", axis=1)

        sc = pyspark.SparkContext()
        transition_counts_rdd = MarkovChain.csr_matrix_to_rdd(sc, transition_counts)
        transition_probabilities_rdd = MarkovChain.csr_matrix_to_rdd(sc, transition_probabilities)

        ml = MarkovChain.marginal_likelihood(transition_counts_rdd, transition_probabilities_rdd, [0, 1, 2, 5], 1.0)
        sc.stop()

        print(ml)
