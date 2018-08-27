import numpy as np
import helpers as hs
import metrosampler.sampler as sr
import metrosampler.posterior as pr


class TestSampler(object):

    def test_running_mean(self):
        """Check correctness of the running mean computation."""
        constraints = hs.MockedConstraints()
        distribution = pr.ConstrainedDistribution(constraints)

        x = distribution.get_example()
        cov = np.identity(2)
        sampler = sr.MetroSampler(distribution, x, cov, 0, 0, 0)

        samples, _, _ = sampler.sample(200, 1)
        samples = np.append(samples[:-1], x.reshape(1, len(x)), 0)

        x_mean = np.mean(samples, 0)
        assert abs(x_mean[0] - sampler.x_mean[0]) <= 1.0e-5
        assert abs(x_mean[1] - sampler.x_mean[1]) <= 1.0e-5

    def test_running_covariance(self):
        """Check correctness of the running covariance computation."""
        constraints = hs.MockedConstraints()
        distribution = pr.ConstrainedDistribution(constraints)

        x = distribution.get_example()
        cov = np.identity(2)
        sampler = sr.MetroSampler(distribution, x, cov, 0, 0, 0)

        samples, _, _ = sampler.sample(200, 1)
        samples = np.append(samples[:-1], x.reshape(1, len(x)), 0)

        x_mean = np.mean(samples, 0)
        x2_sum = np.zeros(cov.shape)
        for val in samples:
            x2_sum += np.outer(val, val)

        nlen = samples.shape[0]
        fact = float(nlen - 1.0)
        x_covariance = (x2_sum - float(nlen)*np.outer(x_mean, x_mean)) / fact

        assert abs(x_covariance[0, 0] - sampler.x_covariance[0, 0]) <= 1.0e-5
        assert abs(x_covariance[1, 1] - sampler.x_covariance[1, 1]) <= 1.0e-5
        assert abs(x_covariance[0, 1] - sampler.x_covariance[0, 1]) <= 1.0e-5





