import checks
import numpy as np


class Distribution:
    """Define the interface for a distribution."""

    def __init__(self):
        pass

    def get_example(self):
        raise NotImplementedError('Abstract method to be implemented')

    def eval(self, x):
        """
        Evaluate the probability density of a state.

        :param x: the state to be evaluated (numpy.ndarray, must be a vector)
        :return: the probability density of the input state
        """
        raise NotImplementedError('Abstract method to be implemented')


class ConstrainedDistribution(Distribution):
    """Define a uniform distribution with constraints on the hypercube."""

    def __init__(self, constraints):
        """
        Initialize the constrained distribution.

        :param constraints: the constraints imposed on the distribution
            (constraints.Constraint)
        """
        self.constraints = constraints
        self.ndim = constraints.get_ndim()

    def get_example(self):
        """Return a feasible state of the distribution."""
        return np.array(self.constraints.get_example())

    def eval(self, x):
        """
        Evaluate whether the input state is in the feasible region or not.

        :param x: the state to be evaluated (numpy.ndarray, must be a vector
            of valid size)
        :return: 1 if the state lies in the feasible region, 0 otherwise
        :raise: ValueError if the dimension of the input state differs from
            the dimension of the underlying constraints object
        """
        # Input must be valid
        checks.check_vector_validity(x)
        checks.check_vector_size(x, self.ndim)

        # x must belong to the n-dimensional hypercube
        for val in x:
            if val < 0 or val > 1.0:
                return 0.0

        valid_constraints = self.constraints.apply(x.tolist())
        if valid_constraints:
            return 1.0

        return 0.0
