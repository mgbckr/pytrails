from unittest import TestCase
import pyspark
from sklearn.preprocessing import normalize
import pytrails.sparktrails.matrixutils as utils
from pytrails.sparktrails.markovchain import MarkovChain


class TestSparkMatrixUtils(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sc = pyspark.SparkContext()

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()

    def test_textfile_rdd_to_tuple_matrix_rdd(self):
        rawtext_rdd = self.sc.parallelize([
            "0\t0;1.0,1;2.0,2;3.0",
            "1\t0;4.0,1;5.0,2;6.0",
            "2\t0;7.0,1;8.0,2;9.0"
        ])
        transition_counts_rdd = utils.textfile_rdd_to_tuple_matrix_rdd(rawtext_rdd)
        print(transition_counts_rdd.collect())

    def test_textfile_rdd_max_columns(self):
        rawtext_rdd = self.sc.parallelize([
            "0\t0;1.0,1;2.0,2;3.0",
            "1\t0;4.0,1;5.0,2;6.0",
            "2\t0;7.0,1;8.0,2;9.0"
        ])
        columns = utils.textfile_rdd_max_columns(rawtext_rdd)
        print(columns)

    def test_marginal_likelihood_from_rdds(self):
        rawtext_rdd = self.sc.parallelize([
            "0\t0;1.0,1;2.0,2;3.0",
            "1\t0;4.0,1;5.0,2;6.0",
            "2\t0;7.0,1;8.0,2;9.0"
        ])
        transition_counts_rdd = utils.textfile_rdd_to_csr_matrix_rdd(rawtext_rdd, 3)
        print(transition_counts_rdd.mapValues(lambda x: x.todense()).collect())

        transition_probabilities_rdd = transition_counts_rdd. \
            mapValues(lambda line: normalize(line, "l1", axis=1))

        ml = MarkovChain.marginal_likelihood(
            transition_counts_rdd,
            transition_probabilities_rdd,
            [0, 1, 2, 5],
            1.0)

        print(ml)

