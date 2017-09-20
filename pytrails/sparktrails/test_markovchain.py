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

    def test_marginal_likelihood_zeros(self):

        transition_counts = csr_matrix([[1, 2, 3], [0, 0, 0], [7, 8, 9]], dtype=np.float64)
        transition_probabilities = normalize(transition_counts, "l1", axis=1)

        sc = pyspark.SparkContext()
        transition_counts_rdd = MarkovChain.csr_matrix_to_rdd(sc, transition_counts)
        transition_probabilities_rdd = MarkovChain.csr_matrix_to_rdd(sc, transition_probabilities)

        ml = MarkovChain.marginal_likelihood(transition_counts_rdd, transition_probabilities_rdd, [0, 1, 2, 5], 1.0)
        sc.stop()

        print(ml)

    def test_marginal_likelihood_zeros2(self):
        transition_counts = csr_matrix([[1, 2, 3], [0, 0, 0], [7, 8, 9]], dtype=np.float64)
        transition_probabilities = normalize(
            csr_matrix([[1, 2, 3], [1, 0, 0], [7, 8, 9]], dtype=np.float64), "l1", axis=1)

        sc = pyspark.SparkContext()
        transition_counts_rdd = MarkovChain.csr_matrix_to_rdd(sc, transition_counts)
        transition_probabilities_rdd = MarkovChain.csr_matrix_to_rdd(sc, transition_probabilities)

        ml = MarkovChain.marginal_likelihood(transition_counts_rdd, transition_probabilities_rdd, [0, 1, 2, 5], 1.0)
        sc.stop()

        print(ml)

    def test_marginal_likelihood_zeros3(self):
        transition_counts = csr_matrix([[1, 2, 3], [1, 0, 0], [7, 8, 9]], dtype=np.float64)
        transition_probabilities = normalize(
            csr_matrix([[1, 2, 3], [0, 0, 0], [7, 8, 9]], dtype=np.float64), "l1", axis=1)

        sc = pyspark.SparkContext()
        transition_counts_rdd = MarkovChain.csr_matrix_to_rdd(sc, transition_counts)
        transition_probabilities_rdd = MarkovChain.csr_matrix_to_rdd(sc, transition_probabilities)

        ml = MarkovChain.marginal_likelihood(transition_counts_rdd, transition_probabilities_rdd, [0, 1, 2, 5], 1.0)
        sc.stop()

        print(ml)


    def test_marginal_likelihood_from_rdds(self):
        sc = pyspark.SparkContext()
        rawtext_rdd = sc.parallelize([
            "0\t0;1.0,1;2.0,2;3.0",
            "1\t0;4.0,1;5.0,2;6.0",
            "2\t0;7.0,1;8.0,2;9.0"
        ])
        transition_counts_rdd = MarkovChain.parse_hdfs_textfile(rawtext_rdd)
        print(transition_counts_rdd.mapValues(lambda x: x.todense()).collect())
        transition_probabilities_rdd = transition_counts_rdd. \
            mapValues(lambda line: normalize(line, "l1", axis=1))
        ml = MarkovChain.marginal_likelihood(transition_counts_rdd, transition_probabilities_rdd, [0, 1, 2, 5], 1.0)
        print(ml)
        sc.stop()
