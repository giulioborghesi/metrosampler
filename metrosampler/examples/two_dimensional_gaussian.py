import math
import numpy as np
import metrosampler.checks as sc
import metrosampler.sampler as sr
import metrosampler.posterior as sp
import matplotlib as mpl

mpl.use('TkAgg')
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

    # Samplers parameter
    t0 = 1000
    tb = 50000

    # Generate samples
    x0 = posterior.get_example()
    sampler = sr.MetroSampler(posterior, x0, covariance, 200, t0, tb, 0.1)
    vals, _, _ = sampler.sample(10000, 100)

    # Generate scatter plot
    plt.scatter(vals[:, 0], vals[:, 1], s = 0.2)
    plt.show()


if __name__ == '__main__':
    generate_samples()
