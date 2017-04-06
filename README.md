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
  - fitting routine
    - take in a list of samples and EM algorithm our way to a best fit distribution?
    - will need to profile this, possibly optimize. Could easily get very expensive.

### distrib_sanityCheck

  - write posteriors to disk in a recoverable way (pkl? npy?)
  - overlay expected scaling (parameterized by rho2) at various values of eta2/rho2
  - include effects of sampling errors

--------------------------------------------------

## Dependencies

  - optparse
  - numpy
  - scipy (specifically, scipy.optimize.netwon is used to find the zeros of non-linear functions)
  - mpmath (used for high-precision calculation of modified bessel functions of the first kind)
