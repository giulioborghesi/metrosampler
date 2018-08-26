import pytest
import numpy as np
import helpers as hp
import sampler.posterior as sp


class TestPosterior(object):

    def test_get_example(self):
        """Test get_example method."""
        constraints = hp.MockedConstraints()
        distribution = sp.ConstrainedDistribution(constraints)

        x = distribution.get_example()
        assert x[0] == .2
        assert x[1] == .5

    def test_eval_valid_x_size_unfeasible(self):
        """Test eval method with an unfeasible state."""
        constraints = hp.MockedConstraints()
        distribution = sp.ConstrainedDistribution(constraints)

        x = np.array([1.2, .2])
        valid = distribution.eval(x)

        assert abs(valid-0.0) <= 1.0e-5

    def test_eval_valid_x_size_feasible(self):
        """Test eval method with a feasible state."""
        constraints = hp.MockedConstraints()
        distribution = sp.ConstrainedDistribution(constraints)

        x = np.array([.2, .9])
        valid = distribution.eval(x)

        assert abs(valid-1.0) <= 1.0e-5

    def test_eval_invalid_x(self):
        """Test eval method with a state of incompatible size."""
        constraints = hp.MockedConstraints()
        distribution = sp.ConstrainedDistribution(constraints)

        x = np.array([.1, .1, .1])
        with pytest.raises(ValueError):
            distribution.eval(x)
