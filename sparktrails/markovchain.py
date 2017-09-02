import numpy as np
import pyspark
from scipy.sparse import csr_matrix

from hyptrails.markovchain import MarkovChain as HypTrailsMarkovChain


class MarkovChain:
    @staticmethod
    def parse_hdfs_textfile(textfile_rdd, maxid=-1):
        """
        :type sc: pyspark.SparkContext
        :param sc: SparkContext

        :type textfile_rdd: pyspark.RDD
        :param textfile_rdd: the raw textfile with sparse matrix entries in the format "<sourceid>\t[<targetid>;<probability>][,<targetid>;<probability>]+
        """
        preparse = textfile_rdd \
            .map(lambda line: line.split("\t")) \
            .map(lambda parts: (
            int(parts[0]),
            np.array([[0] + pos_value.split(";") for pos_value in parts[1].split(",")], dtype=np.float64)))
        if maxid < 0:
            maxid = max(preparse.flatMap(lambda x: [x[0]] + list(x[1][:, 1])).distinct().collect())
        return preparse \
            .mapValues(lambda entries: csr_matrix((entries[:, 2], (entries[:, 0], entries[:, 1])), shape=(1, maxid)))

    @staticmethod
    def csr_matrix_to_rdd(sc, matrix, num_slices=None):
        """
        :type sc: pyspark.SparkContext
        :param sc: SparkContext

        :type matrix: csr_matrix
        :param matrix: the matrix to parallelize

        :type num_slices: int
        :param num_slices: slices to use when parallelizing the matrix
        """
        matrix_rows = [(i, matrix[i, :]) for i in range(matrix.shape[0])]
        return sc.parallelize(matrix_rows, num_slices)

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
        def combine(transition_counts_row, pseudo_counts_row):
            return np.array([
                HypTrailsMarkovChain.marginal_likelihood(
                    transition_counts_row,
                    pseudo_counts_row * cf,
                    smoothing)
                for cf in concentration_factors])
        # do the actual calculations
        return aligned\
            .mapValues(lambda e: combine(e[0], e[1]))\
            .values()\
            .reduce(lambda a, b: a + b)
