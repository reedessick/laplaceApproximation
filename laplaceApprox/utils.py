__description__ = "a module that houses convenient functions for the Laplace approximation of distributions of BayesFactors along with marginalization"
__author__ = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import numpy as np
import mpmath ### used for high-precision computation of modified bessel function
from scipy.optimize import newton ### contains nonlinear root finders

#-------------------------------------------------

#--- general utility functions

def sumLogs( arrayLike ):
    '''
    return ln(sum(np.exp(arrayLike))) to high accuracy
    '''
    maxVal = np.max(arrayLike)
    return maxVal + np.log(np.sum(np.exp(np.array(arrayLike)-maxVal)))

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

def lnBSN_to_rho2( lnBSN, params, f_tol=1e-10 ):
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
        return newton(func, 2*lnBSN, args=(lnBSN,), fprime=fprime, fprime2=fprime2, tol=f_tol)

    else:
        ans = [newton(func, 2*lnbsn, args=(lnbsn,), fprime=fprime, fprime2=fprime2, tol=f_tol) for lnbsn in lnBSN]
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

def lnProb_rhoA2rhoB2_given_rhoA2orhoB2o( rhoA2, rhoB2, rhoA2o, rhoB2o ):
    '''
    return ln( p(rhoA2,rhoB2|rhoA2o,rhoB2o) )

    NOTE: essentially 2 independent chi2 distributions
    '''
    if isinstance(rhoA2, (int,float)) and isinstance(rhoB2, (int,float)) and isinstance(rhoA2o, (int,float)) and isinstance(rhoB2o, (int,float)):
        return np.log(0.25) - 0.5*(rhoA2 + rhoB2 + rhoA2o + rhoB2o) \
            + float(mpmath.log(mpmath.besseli(0, (rhoA2*rhoA2o)**0.5))) \
            + float(mpmath.log(mpmath.besseli(0, (rhoB2*rhoB2o)**0.5)))
    else:
        N = len(rhoA2)
        assert N==len(rhoB2), 'rhoA2 and rhoB2 do not have the same length'
        assert N==len(rhoA2o), 'rhoA2 and rhoA2o do not have the same length'
        assert N==len(rhoB2o), 'rhoA2 and rhoB2o do not have the same length'

        log25 = np.log(0.25)
        ans = [log25 - 0.5*(rhoa2 + rhob2 + rhoa2o + rhob2o) \
            + float(mpmath.log(mpmath.besseli(0, (rhoa2*rhoa2o)**0.5))) \
            + float(mpmath.log(mpmath.besseli(0, (rhob2*rhob2o)**0.5))) \
            for rhoa2, rhob2, rhoa2o, rhob2o in zip(rhoA2, rhoB2, rhoA2o, rhoB2o)
        ]
        if isinstance(rhoA2, np.ndarray) or isinstance(rhoB2, np.ndarray) or isinstance(rhoA2o, np.ndarray) or isinstance(rhoB2o, np.ndarray):
            ans = np.array(ans)

        return ans

def lnProb_rho2eta2_given_rhoA2orhoB2o( rho2, eta2, rhoA2o, rhoB2o ):
    '''
    return ln( p(rho2, eta2 | rhoA2o, rhoB2o) )

    NOTE: 2 independent chi2 distribution with jacobian for transformation of variables
    '''
    rhoA2, rhoB2 = rho2eta2_to_rhoA2rhoB2( rho2, eta2 )

    ### note, there are two solutions here (interchange of A<->B), so we include them both
    ab = lnProb_rhoA2rhoB2_given_rhoA2orhoB2o( rhoA2, rhoB2, rhoA2o, rhoB2o )
    ba = lnProb_rhoA2rhoB2_given_rhoA2orhoB2o( rhoB2, rhoA2, rhoA2o, rhoB2o )

    return np.log(rho2) - 0.5*np.log(rho2**2 - 4*eta2*rho2) \
        + sumLogs( [ab, ba] )    

def lnProb_lnBSNlnBCI_given_rhoA2orhoB2o( lnBSN, lnBCI, rhoA2o, rhoB2o, params, f_tol=1e-10 ):
    '''
    return ln( p(lnBSN, lnBCI | rhoA2o, rhoB2o) )

    NOTE: 2 independent chi2 distributions with jacobian for transformation of variables
    '''
    rho2 = lnBSN_to_rho2( lnBSN, params, f_tol=f_tol )
    eta2 = lnBCI_to_eta2( lnBCI, params )

    return np.log(rho2) - np.log(params.c_bsn + 0.5*rho2) \
        + np.log(eta2) - np.log(params.c_bci) \
        + lnProb_rho2eta2_given_rhoA2orhoB2o( rho2, eta2, rhoA2o, rhoB2o )

#--- marginal distributions

def lnProb_rhoA2_given_rhoA2o( rhoA2, rhoA2o ):
    '''
    return ln( p(rhoA2 | rhoA2o)

    NOTE: essentially a chi2 distribution
    '''
    if isinstance(rhoA2, (int,float)) and isinstance(rhoA2o, (int,float)):
        return np.log(0.5) - 0.5*(rhoA2 + rhoA2o) \
            + float(mpmath.log(mpmath.besseli(0, (rhoA2*rhoA2o)**0.5)))
    else:
        assert len(rhoA2)==len(rhoA2o), 'rhoA2 and rhoA2o do not have the same length'

        log5 = np.log(0.5)
        ans = [log5 - 0.5*(rhoa2 + rhoa2o) \
            + float(mpmath.log(mpmath.besseli(0, (rhoa2*rhoa2o)**0.5))) \
            for rhoa2, rhoa2o in zip(rhoA2, rhoA2o)
        ]
        if isinstance(rhoA2, np.ndarray) or isinstance(rhoA2o, np.ndarray):
            ans = np.array(ans)

        return ans

def lnProb_rhoB2_given_rhoB2o( rhoB2, rhoB2o ):
    '''
    return ln( p(rhoB2 | rhoB2o)

    NOTE: essentially a chi2 distribution
    '''
    return lnProb_rhoA2_given_rhoA2o( rhoB2, rhoB2o )

def lnProb_rho2_given_rhoA2orhoB2o( rho2, rhoA2o, rhoB2o ):
    '''
    return ln( p(rho2 | rhoA2o, rhoB2o)
    '''
    raise NotImplementedError

def lnProb_eta2_given_rhoA2orhoB2o( eta2, rhoA2o, rhoB2o ):
    '''
    return ln( p(eta2 | rhoA2o, rhoB2o)
    '''
    raise NotImplementedError

def lnProb_lnBSN_given_rhoA2orhoB2o( lnBSN, rhoA2o, rhoB2o, params, f_tol=1e-10 ):
    '''
    return ln( p(lnBSN | rhoA2o, rhoB2o)
    '''
    raise NotImplementedError

def lnProb_lnBCI_given_rhoA2orhoB2o( lnBCI, rhoA2o, rhoB2o, params ):
    '''
    return ln( p(lnBCI | rhoA2o, rhoB2o)
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
