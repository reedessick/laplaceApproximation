# laplaceApproximation

A module that approximates the expected distributions of bayes factors (lnBSN and lnBCI) based on the Laplace approximation and expected errors due to Gaussian noise fluctuations and sampling errors. 
This formalism should allow us to determine the expected distributions for any "model," by which I mean a joint prior distribution on how the actual signal-to-noise ratios are distributed in each detector.

--------------------------------------------------

## Dependencies

  - optparse
  - numpy
  - scipy (specifically, scipy.optimize.netwon is used to find the zeros of non-linear functions)
  - mpmath (used for high-precision calculation of modified bessel functions of the first kind)
