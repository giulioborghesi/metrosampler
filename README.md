# metrosampler 
[![Build Status](https://travis-ci.org/giulioborghesi/Adaptive-Metropolis-Sampler.svg?branch=master)](https://travis-ci.org/giulioborghesi/Adaptive-Metropolis-Sampler)

An adaptive implementation of the Metropolis algorithm to sample distributions in high-dimensional space

## Description
This repository provides a Python implementation of the adaptive Metropolis algorithm described in the following article:

> H. Haario, E. Saksman, J. Tamminen, An adaptive Metropolis algorithm, Bernoulli 7(2), 2001, pp. 223-242

The sampling algorithm is implemented by class `MetroSampler`. This class provides a minimal API with a single method, `sample`, which takes 
as arguments the number of samples to generate and the number of steps by which the Markov chain is advanced before generating a sample. 
A minimal usage example is provided below:

    sampler = sr.MetroSampler(posterior, x0, covariance)
    vals, accepted, total = sampler.sample(2000, 100)
    
`sample` returns the requested number of samples as a NumPy array. It also returns the numbers of accepted and attempted steps in running 
the Markov chain. It should be noted that, in the adaptive Metropolis algorithm, the process from which the samples are generated is no longer 
Markovian: for simplicitly, however, this technical detail will be ignored in the following.

To create instances of `MetroSampler`, the sampling distribution must be specified, as well as the state from which the Markov process is 
started and the covariance matrix for the multivariate Gaussian distribution used internally to generate the next states in the process. 
The sampling distribution must be an object of a class that implements the interface defined by the abstract class `Distribution`, whose 
definition can be found in [sampler/posterior.py](https://github.com/giulioborghesi/metrosampler/blob/master/metrosampler/posterior.py).

## Installation

MetroSampler can be installed using `pip` after having cloned the repository to your computer:

    git clone https://github.com/giulioborghesi/metrosampler.git
    cd metrosampler
    pip install .

## Dependencies

MetroSampler is tested on Python 2.7 and depends on NumPy, SciPy, pytest and MatplotLib (see requirements.txt for version information).

