import checks
import numpy as np


def sample_gauss(covariance, samples=1):
    """
    Generate samples from a multivariate Gaussian distribution.

    :param covariance: distribution covariance (numpy.ndarray, must be
        a square matrix)
    :param samples: number of samples to generate (int)
    :return: array of samples from a multivariate Gaussian distribution
    """
    # Input must be valid
    checks.check_matrix_validity(covariance)

    ndim = covariance.shape[0]
    samples = samples if samples > 0 else 1
    return np.random.multivariate_normal(np.zeros(ndim), covariance, samples)


class AdaptiveMetropolisSampler:
    """
    Generate samples from a distribution using the Metropolis algorithm.

    The details of the algorithm are described in the following article:

        - H. Haairo, E. Saksman, and J. Tamminen, An adaptive metropolis
          algorithm, Bernoulli 7(2), 2001, pp. 223-242
    """

    def __init__(self, posterior, x_initial, covariance_initial,
                 update_freq=200, t0=1000, tb=10000, gamma=1.0, eps=1.0e-6):
        """
        Initialize the sampler.

        :param posterior: distribution to sample from (posterior.Distribution)
        :param x_initial: a state of the sampling distribution with nonzero
            probability (numpy.ndarray, must be a vector)
        :param covariance_initial: the initial covariance matrix used by the
            proposal distribution (numpy.ndarray, must be a square matrix)
        :param update_freq: frequency at which the covariance is updated (int)
        :param t0: steps before the learnt covariance matrix is used (int)
        :param tb: length of burnin period (int)
        :param gamma: scaling factor for the transitions step size (float)
        :param eps: small quantity to avoid a singularity in the covariance
            matrix of the proposal distribution (float)
        """
        # Input must be valid
        checks.check_distribution_validity(posterior)
        checks.check_vector_matrix_validity(x_initial, covariance_initial)

        self.posterior = posterior
        self.covariance = covariance_initial

        self.x_last = x_initial
        self.x_last_probability = posterior.eval(x_initial)
        self.x_mean = np.zeros(x_initial.shape)
        self.x_covariance = np.zeros(covariance_initial.shape)

        self.t0 = t0
        self.eps = eps
        self.niter = 0

        self.candidates = None
        self.update_freq = update_freq if update_freq > 0 else 1
        self.sd = (2.7 ** 2) * gamma / float(x_initial.shape[0])

        # Run Markov chain for tb steps before starting to sample
        for _ in range(tb):
            self._step()

    def _update_running_mean(self):
        """Update the state vector mean."""
        diff = self.x_last - self.x_mean
        self.x_mean = self.x_mean + diff / float(1.0 + self.niter)

    def _update_running_covariance(self):
        """Update the empirical covariance matrix."""
        if self.niter == 0:
            return

        diff = self.x_last - self.x_mean
        prod = np.outer(diff, diff) / float(self.niter + 1.0)
        fact = float(self.niter - 1.0) / float(self.niter)
        self.x_covariance = fact * self.x_covariance + prod

    def _update_covariance(self):
        """Update the covariance matrix used to generate candidate states."""
        delta = self.sd * self.eps * np.identity(self.covariance.shape[0])
        self.covariance = self.sd * self.x_covariance + delta
        self.candidates = sample_gauss(self.covariance, self.update_freq)

    def _generate_candidate(self):
        """Generate a new candidate state."""
        if self.candidates is None or self.candidates.size == 0:
            self.candidates = sample_gauss(self.covariance, self.update_freq)

        candidate = self.x_last + self.candidates[0]
        self.candidates = np.delete(self.candidates, 0, 0)
        return candidate

    def _step(self):
        """
        Implements a step of the Metropolis algorithm.

        :return: True if the proposal state is accepted, False otherwise
        """
        # Covariance depends on running mean and must be updated first
        self._update_running_covariance()
        self._update_running_mean()

        self.niter += 1
        x_candidate = self._generate_candidate()
        x_candidate_probability = self.posterior.eval(x_candidate)

        cutoff = np.random.random()
        ratio = x_candidate_probability / self.x_last_probability

        if self.niter % self.update_freq == 0 and self.niter > self.t0:
            self._update_covariance()

        candidate_feasible = ratio >= cutoff
        if candidate_feasible:
            self.x_last = x_candidate
            self.x_last_probability = x_candidate_probability
            return True

        return False

    def sample(self, samples_number=1, sample_every=200):
        """
        Generate samples from the distribution.

        :param samples_number: the number of samples to generate (int)
        :param sample_every: the sampling frequency (int)
        :return: list of samples and numbers of accepted / attempted jumps
        """
        jumps_total = 0
        jumps_accepted = 0

        samples_list = np.array([])
        samples_left = samples_number

        # Ensure sample frequency is at least one
        sample_every = sample_every if sample_every > 0 else 1
        while samples_left > 0:

            for _ in range(sample_every):
                success = self._step()
                if success:
                    jumps_accepted += 1

                jumps_total += 1

            samples_left -= 1
            samples_list = np.append(samples_list, self.x_last)

        reshape_dims = (samples_number, self.x_last.shape[0])
        samples_list = samples_list.reshape(reshape_dims)
        return samples_list, jumps_accepted, jumps_total
