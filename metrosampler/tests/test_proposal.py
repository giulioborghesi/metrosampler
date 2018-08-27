import pytest
import numpy as np
import metrosampler.sampler as sr


class TestProposal(object):

    def test_two_dimensional_gaussian(self):
        """
        Generate samples from multivariate Gaussian and verify that their
        mean is approximately zero.
        """
        covariance = np.identity(2)
        samples = sr.sample_gauss(covariance, 10000)
        samples_mean = np.mean(samples, 0)

        assert abs(samples_mean[0] - 0.) < 1.0e-1
        assert abs(samples_mean[1] - 0.) < 1.0e-1
