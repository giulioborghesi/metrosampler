import os
import argparse
import numpy as np
import metrosampler.sampler as samp
import metrosampler.posterior as post
import metrosampler.constraints as cons


def adjust_gamma(ratio, gamma):
    """
    Tune the scaling parameter for the step size in Markov chain stepper.

    Inspired by implementation of adaptive Metropolis algorithm in PyMC3
    """
    # Do nothing if the ratio is acceptable
    if ratio > 0.20 and ratio < 0.35:
        return gamma

    # Adjust gamma based on current ratio of accepted / total jumps
    if ratio < 0.05:
        gamma *= 0.10
    elif ratio < 0.10:
        gamma *= 0.15
    elif ratio < 0.20:
        gamma *= 0.5
    elif ratio > 0.95:
        gamma *= 10.0
    elif ratio > 0.75:
        gamma *= 6.0
    elif ratio > 0.50:
        gamma *= 3.0
    elif ratio > 0.35:
        gamma *= 2.0

    return gamma    


def gendist():

    # Arguments descriptions
    descA = 'Generate samples from high-dimensional uniform distribution'
    descI = 'path to file with constraints specification'
    descO = 'path to file where sample will be saved'
    descS = 'number of samples to generate'

    # Initialize parser and parse arguments
    parser = argparse.ArgumentParser(description=descA)
    parser.add_argument('inpfile', help=descI)
    parser.add_argument('outfile', help=descO)
    parser.add_argument('samples', help=descS, type=int)
    args = parser.parse_args()

    # Check that constraints file exists
    input_file = args.inpfile
    if not os.path.isfile(input_file):
        print 'Error: the specified input file does not exist !!!'
        exit(0) 

    # Check that output file can be written
    try:
       with open(args.outfile, 'w') as f:
           pass
    except IOError as err:
       print 'Cannot write to output file: ', err.errno, ',', err.strerror
       exit(0)

    # Create constrained distribution
    constraints = cons.Constraint(input_file)
    posterior = post.ConstrainedDistribution(constraints)

    # Initial state of Markov chain and initial covariance matrix
    x0 = posterior.get_example()
    cov0 = np.identity(x0.shape[0])

    # Initialize sampler parameters
    t0 = 1000 
    tb = 10000
    ratio = .0
    gamma = .001
    steps = 0

    # Tune the sampler step size
    print "Adjusting the step size...\n"
    while steps < 5 and (ratio < .20 or ratio > .35):
         sampler = samp.MetroSampler(posterior, x0, cov0, 200, t0, tb, gamma)
         vals, accepted, total = sampler.sample(2000, 10)
         print 'With gamma = %f, the number of accepted and total ' \
               'samples is %d %d' % (gamma, accepted, total) 
         ratio = float(accepted) / float(total) 
         gamma = adjust_gamma(ratio, gamma)
         steps += 1

    # Start sampling
    print '\nThe optimal step size is gamma = %f' % gamma, '. Start sampling...'
    sampler = samp.MetroSampler(posterior, x0, cov0, 200, t0, tb*5, gamma) 
    vals, accepted, total = sampler.sample(args.samples, 100)

    # Store samples to file
    print 'Sampling completed. The number of accepted and total ' \
          'samples is: %d %d' % (accepted, total), '. Storing data to file..'
    np.savetxt(args.outfile, vals, delimiter='  ',fmt='%1.4e')

