import numpy as np
import posterior


def check_distribution_validity(distribution):
    """Raise an assertion if distribution is not valid."""
    if not isinstance(distribution, posterior.Distribution):
        raise ValueError('Error: input is not a valid distribution')


def check_vector_validity(x_vector):
    """Raise an assertion if x_vector is not a NumPy vector."""
    if not isinstance(x_vector, np.ndarray):
        raise ValueError('Error: input is not a NumPy ndarray')

    if len(x_vector.shape) != 1:
        raise ValueError('Error: input is not a vector')


def check_vector_size(x_vector, size):
    """Raise an assertion if x_vector size differs from size."""
    if x_vector.shape[0] != size:
        raise ValueError('Error: input vector size is not valid')


def check_matrix_validity(x_matrix):
    """Raise an assertion is x_matrix is not a symmetric NumPy matrix."""
    if not isinstance(x_matrix, np.ndarray):
        raise ValueError('Error: input is not a NumPy ndarray')

    if len(x_matrix.shape) != 2:
        raise ValueError('Error: input is not a matrix')

    if x_matrix.shape[0] != x_matrix.shape[1]:
        raise ValueError('Error: input is not a symmetric matrix')


def check_vector_matrix_validity(x_vector, x_matrix):
    """Raise an assertion if x_vector, x_matrix have different dimensions."""
    check_vector_validity(x_vector)
    check_matrix_validity(x_matrix)

    if x_vector.shape[0] != x_matrix.shape[0]:
        raise ValueError('Error: inputs have incompatible dimensions')
