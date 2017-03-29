__description__ = "a module that houses convenient functions for the Laplace approximation of distributions of BayesFactors along with marginalization"
__author__ = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import numpy as np
import scipy
import scipy.optimize ### contains nonlinear root finders

#-------------------------------------------------

class BayesMapParams(object):
    '''
    a special-built storate structure for parameters defining our expected scalings of BayesFactors
    '''

    def __init__(self, c_bsn=1.0, g_bsn=1.0, c_bci=1.0, g_bci=1.0):
        self.c_bsn = c_bsn
        self.g_bsn = g_bsn
        self.c_bci = c_bci
        self.g_bci = g_bci
    
#-------------------------------------------------

#--- mappings between bayes factors and SNR-like statistics

def lnBSN_to_rho2( lnBSN, params, f_tol=1e-14 ):
    '''
    computes rho2(lnBSN)
    
    WARNING: this solves a transcendental equation.
    You may need to be careful about convergence 
    and computational efficiency if it is called repeatedly
    '''
    func = lambda rho2, lnbsn: rho2_to_lnBSN(rho2, params)-lnbsn
    fprime = lambda rho2, *args: 0.5 + params.c_bsn/rho2
    fprime2 = lambda rho2, *args: -params.c_bsn/rho2**2

    ### delegate to standard nonlinear root finder
    if isinstance(lnBSN, (float, int)): ### only one thing to do
        return scipy.optimize.newton(func, 2*lnBSN, args=(lnBSN,), fprime=fprime, fprime2=fprime2, tol=f_tol)

    else:
        ans = [scipy.optimize.newton(func, 2*lnbsn, args=(lnbsn,), fprime=fprime, fprime2=fprime2, tol=f_tol) for lnbsn in lnBSN]
        if isinstance(lnBSN, np.ndarray):
            ans = np.array(ans)
        return ans

def rho2_to_lnBSN( rho2, params ):
    '''
    computes lnBSN(rho2)
    '''
    return 0.5*rho2 + params.c_bsn*np.log(rho2) + params.g_bsn

def lnBCI_to_eta2( lnBCI, params ):
    '''
    computes eta2(lnBCI)
    '''
    return np.exp( (lnBCI-params.g_bci)/params.c_bci )

def eta2_to_lnBCI( eta2, params ):
    '''
    computes lnBCI(eta2)
    '''
    return params.c_bci*np.log(eta2) + params.g_bci 

#-------------------------------------------------

#--- mappings between single-IFO statistics and "coherent" SNR-like statistics

def rho2eta2_to_rhoA2rhoB2( rho2, eta2 ):
    '''
    converts rho2, eta2 -> rhoA2, rhoB2

    NOTE: this always returns the larger of the single-IFO BayesFactors first.
    You may want to actually sample from both orderings when using the result
    '''
    det = 0.5*(rho2**2 - 4*eta2*rho2)**0.5
    const = 0.5*rho2
    return const + det, const - det

def rhoA2rhoB2_to_rho2eta2( rhoA2, rhoB2 ):
    '''
    converts rhoA2, rhoB2 -> rho2, eta2
    '''
    rho2 = rhoA2+rhoB2
    return rho2, rhoA2*rhoB2/rho2

#-------------------------------------------------

#--- define probabilty distributions given rhoA2o, rhoB2o

def lnProb_rhoA2rhoB2_given_rhoA2orhoB2o( rhoA2o, rhoB2o ):
    '''
    return ln( p(rhoA2,rhoB2|rhoA2o,rhoB2o) )

    NOTE: essentially 2 independent chi2 distributions
    '''
    raise NotImplementedError

def lnProb_rho2eta2_given_rhoA2orhoB2o( rhoA2o, rhoB2o ):
    '''
    return ln( p(rho2, eta2 | rhoA2o, rhoB2o) )

    NOTE: 2 independent chi2 distribution with jacobian for transformation of variables
    '''
    raise NotImplementedError

def lnProb_lnBSNlnBCI_given_rhoA2orhoB2o( rhoA2o, rhoB2o ):
    '''
    return ln( p(lnBSN, lnBCI | rhoA2o, rhoB2o) )

    NOTE: 2 independent chi2 distributions with jacobian for transformation of variables
    '''
    raise NotImplementedError

#-------------------------------------------------

#--- define marginalization routines

'''
Define marignalization routines
    - marginalize over sampling errors
        - direct estimation of the integral
        - spectral evaluation of the convolution
        - importance sampling
    - marginalize over a model distribution of rhoA2o, rhoB2o
        - direct estimation of the integral
        - importance sampling
'''
