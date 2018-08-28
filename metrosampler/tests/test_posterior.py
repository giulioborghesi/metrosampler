import pytest
import numpy as np
import helpers as hp
import metrosampler.posterior as sp
import metrosampler.constraints as cs


class TestPosterior(object):

    def test_get_example(self):
        """Test get_example method."""
        constraints = hp.MockedConstraints()
        distribution = sp.ConstrainedDistribution(constraints)

        x = distribution.get_example()
        assert x[0] == .2
        assert x[1] == .5

    def test_prob_valid_x_size_unfeasible(self):
        """Test prob method with an unfeasible state."""
        constraints = hp.MockedConstraints()
        distribution = sp.ConstrainedDistribution(constraints)

        x = np.array([1.2, .2])
        valid = distribution.prob(x)

        assert abs(valid-0.0) <= 1.0e-5

    def test_prob_valid_x_size_feasible(self):
        """Test prob method with a feasible state."""
        constraints = hp.MockedConstraints()
        distribution = sp.ConstrainedDistribution(constraints)

        x = np.array([.2, .9])
        valid = distribution.prob(x)

        assert abs(valid-1.0) <= 1.0e-5

    def test_prob_invalid_x(self):
        """Test prob method with a state of incompatible size."""
        constraints = hp.MockedConstraints()
        distribution = sp.ConstrainedDistribution(constraints)

        x = np.array([.1, .1, .1])
        with pytest.raises(ValueError):
            distribution.prob(x)

    def test_samples_are_valid(self):
        """Verify that the generated samples are valid."""
        constraints = cs.Constraint('metrosampler/tests/Data/alloy.txt')

        x = np.loadtxt('metrosampler/tests/Data/alloy_samples.txt',dtype=float)
        for i in range(x.shape[0]):
            valid = constraints.apply(x[i,:].T)
            assert valid
