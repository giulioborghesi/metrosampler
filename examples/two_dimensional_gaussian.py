import sys
import math
sys.path.append('../')

import numpy as np
import sampler.checks as sc
import sampler.sampler as sr
import sampler.posterior as sp
import matplotlib.pyplot as plt


class TwoDimensionalGaussian(sp.Distribution):
    """Define a two dimensional Gaussian distribution"""

    def __init__(self):
        self.ndim = 2
        self.inv_covariance = np.array([[0.5, -0.2], [-0.2, 0.5]])

    def get_example(self):
        return np.array([.2, .1])

    def eval(self, x):
        # Input must be valid
        sc.check_vector_validity(x)
        sc.check_vector_size(x, self.ndim)

        prod = np.dot(np.dot(x.T, self.inv_covariance), x)
        return math.exp(-prod / 2.)


def generate_samples():

    # Create distribution and initial covariance matrix for proposal
    posterior = TwoDimensionalGaussian()
    covariance = np.identity(2)

    # Generate samples
    sampler = sr.AdaptiveMetropolisSampler(posterior, posterior.get_example(),
                                           covariance, 200, 1000, 50000, 0.1)
    vals, _, _ = sampler.sample(10000, 100)

    # Generate scatter plot
    plt.scatter(vals[:, 0], vals[:, 1], s = 0.2)
    plt.show()


if __name__ == '__main__':
    generate_samples()
