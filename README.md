# laplaceApproximation

A module that approximates the expected distributions of bayes factors (lnBSN and lnBCI) based on the Laplace approximation and expected errors due to Gaussian noise fluctuations and sampling errors. 
This formalism should allow us to determine the expected distributions for any "model," by which I mean a joint prior distribution on how the actual signal-to-noise ratios are distributed in each detector.

## To Do

### laplaceApprox/utils.py

  - write/test (marginalized) 1D posteriors for 
    - rho2 
    - eta2
    - lnBSN
    - lnBCI
  - write more distributions for sample_rho2Aorho2Bo
    - isotropic
    - malmquist
    - background
  - write marginalization/convolution over sampling errors
    - these are somewhat implemented, but either buggy or subject to numerical errors or very expensive
  - write MCMC to sample distribution
    - could be faster to compute than a grid-based solution (which I'm currently doing)
  - fitting routine
    - take in a list of samples and EM algorithm our way to a best fit distribution?
    - will need to profile this, possibly optimize. Could easily get very expensive.
  - regression of the distributions of rhoA2o, rhoB2o given lnBSN, lnBCI
    - not sure this is actually interesting...

### distrib_sanityCheck

  - write posteriors to disk in a recoverable way (pkl? npy?)
  - overlay expected scaling (parameterized by rho2) at various values of eta2/rho2
  - include effects of sampling errors
    - some of this is in place, but it should be extended as I debug/write more ways to marginalize

--------------------------------------------------

## Dependencies

  - optparse
  - numpy
  - scipy (specifically, scipy.optimize.netwon is used to find the zeros of non-linear functions)
  - mpmath (used for high-precision calculation of modified bessel functions of the first kind)
