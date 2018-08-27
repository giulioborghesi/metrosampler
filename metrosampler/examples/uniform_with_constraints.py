import numpy as np
import metrosampler.sampler as sr
import metrosampler.posterior as sp
import metrosampler.constraints as sc
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt


def generate_samples():

    # Create distribution and initial covariance matrix for proposal
    constraints = sc.Constraint('./Data/mixture.txt')
    posterior = sp.ConstrainedDistribution(constraints)
    covariance = np.identity(2)

    # Sampler's parameters
    t0 = 1000
    tb = 50000

    # Generate samples
    x0 = posterior.get_example()
    sampler = sr.MetroSampler(posterior, x0, covariance, 200, t0, tb, 0.1)
    vals, accepted, total = sampler.sample(4000, 200)

    # Generate scatter plot
    plt.scatter(vals[:, 0], vals[:, 1], s = 0.2)
    plt.show()


if __name__ == '__main__':
    generate_samples()
