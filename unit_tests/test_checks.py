import pytest
import numpy as np
import sampler.checks as checks


class TestChecks():

    def test_vector_validity_not_ndarray(self):
        x = [i for i in range(5)]
        with pytest.raises(ValueError):
            checks.check_vector_validity(x)

    def test_vector_validity_not_vector(self):
        x = np.identity(5)
        with pytest.raises(ValueError):
            checks.check_vector_validity(x)

    def test_vector_validity_valid_vector(self):
        x = np.array([i for i in range(5)])
        checks.check_vector_validity(x)

    def test_matrix_validity_not_ndarray(self):
        mat = [[i, j] for i in range(5) for j in range(5)]
        with pytest.raises(ValueError):
            checks.check_matrix_validity(mat)

    def test_matrix_validity_not_matrix(self):
        mat = np.array([i for i in range(5)])
        with pytest.raises(ValueError):
            checks.check_matrix_validity(mat)
