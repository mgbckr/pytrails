from scipy.sparse import csr_matrix


def text_to_tuple_row(
        row_string,
        row_index_separator="\t",
        column_separator=",",
        column_index_separator=";"):

    row_split = row_string.split(row_index_separator)
    row_index = int(row_split[0])
    row_entry_strings = [entry.split(column_index_separator)
                         for entry in row_split[1].split(column_separator)]
    row_entries = [(int(entry[0]), float(entry[1]))
                   for entry in row_entry_strings]

    return row_index, row_entries


def textfile_rdd_to_tuple_matrix_rdd(
        textfile_rdd,
        row_index_separator="\t",
        column_separator=",",
        column_index_separator=";"):

    return textfile_rdd.map(lambda line:
        text_to_tuple_row(line, row_index_separator, column_separator, column_index_separator))


def textfile_rdd_max_columns(
        textfile_rdd,
        row_index_separator="\t",
        column_separator=",",
        column_index_separator=";"):
    return textfile_rdd_to_tuple_matrix_rdd(
            textfile_rdd,
            row_index_separator,
            column_separator,
            column_index_separator).\
        map(lambda e: max([column for column, value in e[1]])).\
        max() + 1


def textfile_rdd_to_csr_matrix_rdd(
        textfile_rdd,
        number_of_columns,
        row_index_separator="\t",
        column_separator=",",
        column_index_separator=";"):

    def parse_row(row_string):
        row_index, tuple_row = text_to_tuple_row(
            row_string,
            row_index_separator,
            column_separator,
            column_index_separator)

        column_indexes = [entry[0] for entry in tuple_row]
        column_values = [entry[1] for entry in tuple_row]

        row = csr_matrix(
            (column_values, column_indexes, [0, len(column_values)]),
            shape=(1, number_of_columns))

        return row_index, row

    return textfile_rdd.map(parse_row)


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
