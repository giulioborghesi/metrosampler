import sys
sys.path.append('../')

import numpy as np
import sampler.sampler as sr
import sampler.posterior as sp
import sampler.constraints as sc
import matplotlib.pyplot as plt


def generate_samples():

    # Create distribution and initial covariance matrix for proposal
    constraints = sc.Constraint('./Data/mixture.txt')
    posterior = sp.ConstrainedDistribution(constraints)
    covariance = np.identity(2)

    # Generate samples
    sampler = sr.AdaptiveMetropolisSampler(posterior, posterior.get_example(),
                                           covariance, 200, 1000, 50000, 0.1)
    vals, accepted, total = sampler.sample(2000, 100)

    # Generate scatter plot
    plt.scatter(vals[:, 0], vals[:, 1], s = 0.2)
    plt.show()


if __name__ == '__main__':
    generate_samples()