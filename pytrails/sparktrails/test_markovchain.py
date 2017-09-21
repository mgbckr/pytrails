from unittest import TestCase
import pyspark
from .markovchain import MarkovChain
from .matrixutils import *
from .livehypotheses import *


class TestSparkMarkovChain(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sc = pyspark.SparkContext()

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()

    def test_marginal_likelihood(self):

        transition_counts = csr_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        transition_probabilities = normalize(transition_counts, "l1", axis=1)

        transition_counts_rdd = csr_matrix_to_rdd(self.sc, transition_counts)
        transition_probabilities_rdd = csr_matrix_to_rdd(self.sc, transition_probabilities)

        ml = MarkovChain.marginal_likelihood(
            transition_counts_rdd,
            transition_probabilities_rdd,
            [0, 1, 2, 5],
            1.0)

        print(ml)

    def test_marginal_likelihood_zeros(self):

        transition_counts = csr_matrix([[1, 2, 3], [0, 0, 0], [7, 8, 9]], dtype=np.float64)
        transition_probabilities = normalize(transition_counts, "l1", axis=1)

        transition_counts_rdd = csr_matrix_to_rdd(self.sc, transition_counts)
        transition_probabilities_rdd = csr_matrix_to_rdd(self.sc, transition_probabilities)

        ml = MarkovChain.marginal_likelihood(
            transition_counts_rdd,
            transition_probabilities_rdd,
            [0, 1, 2, 5],
            1.0)

        print(ml)

    def test_marginal_likelihood_zeros2(self):
        transition_counts = csr_matrix([[1, 2, 3], [0, 0, 0], [7, 8, 9]], dtype=np.float64)
        transition_probabilities = normalize(
            csr_matrix([[1, 2, 3], [1, 0, 0], [7, 8, 9]], dtype=np.float64), "l1", axis=1)

        transition_counts_rdd = csr_matrix_to_rdd(self.sc, transition_counts)
        transition_probabilities_rdd = csr_matrix_to_rdd(self.sc, transition_probabilities)

        ml = MarkovChain.marginal_likelihood(
            transition_counts_rdd,
            transition_probabilities_rdd,
            [0, 1, 2, 5], 1.0)

        print(ml)

    def test_marginal_likelihood_zeros3(self):
        transition_counts = csr_matrix([[1, 2, 3], [1, 0, 0], [7, 8, 9]], dtype=np.float64)
        transition_probabilities = normalize(
            csr_matrix([[1, 2, 3], [0, 0, 0], [7, 8, 9]], dtype=np.float64), "l1", axis=1)

        transition_counts_rdd = csr_matrix_to_rdd(self.sc, transition_counts)
        transition_probabilities_rdd = csr_matrix_to_rdd(self.sc, transition_probabilities)

        ml = MarkovChain.marginal_likelihood(
            transition_counts_rdd,
            transition_probabilities_rdd,
            [0, 1, 2, 5],
            1.0)

        print(ml)

    def test_marginal_likelihood_live_hypothesis(self):

        transition_counts = csr_matrix([[1, 2, 3], [1, 0, 0], [7, 8, 9]], dtype=np.float64)

        transition_counts_rdd = csr_matrix_to_rdd(self.sc, transition_counts)

        transition_counts_with_destination_features = \
            transition_counts_rdd.map(lambda e: (e[0], (e[1], {})))

        ml = MarkovChain.marginal_likelihood_live_hypothesis(
            transition_counts_with_destination_features,
            uniform_no_selftransitions_hypothesis,
            [0, 1, 2, 5],
            1.0)

        print(ml)
