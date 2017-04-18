__description__ = "a module that houses convenient functions for the Laplace approximation of distributions of BayesFactors along with marginalization"
__author__ = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import numpy as np
import mpmath ### used for high-precision computation of modified bessel function
from scipy.optimize import newton ### contains nonlinear root finders
from scipy.special import erfinv ### used to space direct integration points

#-------------------------------------------------

#--- general utility functions

def sumLogs( arrayLike, axis=0 ):
    '''
    return ln(sum(np.exp(arrayLike))) to high accuracy
    '''
    shape = np.shape(arrayLike)
    if len(shape)==1:
        maxVal = np.max(arrayLike)
        return maxVal + np.log(np.sum(np.exp(np.array(arrayLike)-maxVal))) if maxVal>-np.infty else -np.infty

    elif axis!=0:
        ### transpose to ensure axis=0
        ### this just makes all the broadcasting stuff work better below
        axes = range(len(np.shape(arrayLike)))
        axes.insert( 0, axes.pop(axis)) ### there might be a more graceful way to do this...
        arrayLike = np.transpose(arrayLike, axes=axes)    
   
    maxVal = np.max(arrayLike, axis=0)
    ans = maxVal + np.log(np.sum(np.exp(np.array(arrayLike)-maxVal), axis=0))
    ans[maxVal==-np.infty] = -np.infty ### do this to avoid nan's from "np.infty-np.infty"

    return ans 

#-------------------------------------------------

class BayesMapParams(object):
    '''
    a special-built storate structure for parameters defining our expected scalings of BayesFactors

    NOTE: 
        this formalism (in paricular, the definition of eta2) doesn't really make sense unless abs(c_bsn)==abs(c_bci)
        we also expect c_bsn < 0 and c_bci > 0
    however, we do not check for this explicitly or in any way enforce it. The default values do obey this, though.
    '''

    def __init__(self, c_bsn=-1.0, g_bsn=1.0, c_bci=1.0, g_bci=1.0):
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
        lnBSN = np.array(lnBSN)
        return np.array([newton(func, 2*lnbsn, args=(lnbsn,), fprime=fprime, fprime2=fprime2, tol=f_tol) for lnbsn in lnBSN.flatten()]).reshape(lnBSN.shape)

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
    if (rhoA2==0) and (rhoB2==0):
        return 0.0, 0.0

    rho2 = rhoA2+rhoB2
    return rho2, rhoA2*rhoB2/rho2

#-------------------------------------------------

#--- define probabilty distributions given rhoA2o, rhoB2o

def lnProb_rhoA2rhoB2_given_rhoA2orhoB2o( rhoA2, rhoB2, rhoA2o, rhoB2o ):
    '''
    return ln( p(rhoA2,rhoB2|rhoA2o,rhoB2o) )

    NOTE: essentially 2 independent chi2 distributions
    '''
    types = (int,float)
    if isinstance(rhoA2, types) and isinstance(rhoB2, types) \
      and isinstance(rhoA2o, types) and isinstance(rhoB2o, types):
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

        ans[ans!=ans] = -np.infty ### get rid of nans
                              ### FIXME we may want to be smarter about this and 
                              ### prevent nans from showing up in the first place
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

    ans = 0.5*(np.log(rho2) - np.log(rho2 - 4*eta2)) \
        + sumLogs( [ab, ba] )

    ans[ans!=ans] = -np.infty ### get rid of nans
                              ### FIXME we may want to be smarter about this and 
                              ### prevent nans from showing up in the first place
    return ans

def lnProb_lnBSNlnBCI_given_rhoA2orhoB2o( lnBSN, lnBCI, rhoA2o, rhoB2o, params, f_tol=1e-10 ):
    '''
    return ln( p(lnBSN, lnBCI | rhoA2o, rhoB2o) )

    NOTE: 2 independent chi2 distributions with jacobian for transformation of variables
    '''
    rho2 = lnBSN_to_rho2( lnBSN, params, f_tol=f_tol )
    eta2 = lnBCI_to_eta2( lnBCI, params )

    ans = np.log(rho2) - np.log(params.c_bsn + 0.5*rho2) \
        + np.log(eta2) - np.log(params.c_bci) \
        + lnProb_rho2eta2_given_rhoA2orhoB2o( rho2, eta2, rhoA2o, rhoB2o )

    ans[ans!=ans] = -np.infty ### get rid of nans
                              ### FIXME we may want to be smarter about this and 
                              ### prevent nans from showing up in the first place
    return ans

def lnProb_lnlnBSNlnBCI_given_rhoA2orhoB2o( lnlnBSN, lnBCI, rhoA2o, rhoB2o, params, f_tol=1e-10 ):
    '''
    return ln( p(ln(lnBSN), lnBCI | rhoA2o, rhoB2o) )

    NOTE: 2 independent chi2 distributions with jacobian for transformation of variables
    '''
    ans = lnlnBSN + lnProb_lnBSNlnBCI_given_rhoA2orhoB2o(np.exp(lnBSN), lnBCI, rhoA2o, rhoB2o, params, f_tol=f_tol)
    ans[ans!=ans] = -np.infty ### get rid of nans
                              ### FIXME we may want to be smarter about this and 
                              ### prevent nans from showing up in the first place
    return ans

#--- marginal distributions

def lnProb_rhoA2_given_rhoA2o( rhoA2, rhoA2o ):
    '''
    return ln( p(rhoA2 | rhoA2o) )

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
    return ln( p(rhoB2 | rhoB2o) )

    NOTE: essentially a chi2 distribution
    '''
    return lnProb_rhoA2_given_rhoA2o( rhoB2, rhoB2o )

def lnProb_rho2_given_rhoA2orhoB2o( rho2, rhoA2o, rhoB2o ):
    '''
    return ln( p(rho2 | rhoA2o, rhoB2o) )
    '''
    raise NotImplementedError

def lnProb_eta2_given_rhoA2orhoB2o( eta2, rhoA2o, rhoB2o ):
    '''
    return ln( p(eta2 | rhoA2o, rhoB2o) )
    '''
    raise NotImplementedError

def lnProb_lnBSN_given_rhoA2orhoB2o( lnBSN, rhoA2o, rhoB2o, params, f_tol=1e-10 ):
    '''
    return ln( p(lnBSN | rhoA2o, rhoB2o) )
    '''
    raise NotImplementedError

def lnProb_lnBCI_given_rhoA2orhoB2o( lnBCI, rhoA2o, rhoB2o, params ):
    '''
    return ln( p(lnBCI | rhoA2o, rhoB2o) )
    '''
    raise NotImplementedError

def lnProb_lnlnBSN_given_rhoA2orhoB2o( lnlnBSN, rhoA2o, rhoB2o, params ):
    '''
    return ln( p(ln(lnBSN) | rhoA2o, rhoB2o) )
    '''
    raise NotImplementedError

#-------------------------------------------------

#--- define marginalization routines

'''
Define marignalization routines
    - marginalize over a model distribution of rhoA2o, rhoB2o
        - direct estimation of the integral 
            - users probably want to just do this themselves and define their own step size, etc
        - importance sampling 
            - supported through sample_rhoA2orhoB2o
'''

known = {
    'uniform' : [
        'min_rhoA2o', 
        'max_rhoA2o', 
        'min_rhoB2o', 
        'max_rhoB2o',
    ],
    'uniform_rho2o-fixed_eta2o' : [
        'min_rho2o',
        'max_rho2o',
        'eta2oOVERrho2o',
    ],
    'isotropic' : [
    ],
    'malmquist' : [
    ],
    'background' : [
        'nonGauss_fracA',
        'min_rhoA2o',
        'max_rhoA2o',
        'paretoA',
        'nonGauss_fracB',
        'min_rhoB2o',
        'max_rhoB2o',
        'paretoB',
    ],
}

def sample_rhoA2orhoB2o( Nsamp, distrib='uniform', **kwargs ):
    '''
    generates Nsamp samples from the joint distribution of (rhoA2o, rhoB2o)
    the type of distribution is determined by distrib (and kwargs)

    note, distrib must be one of laplaceApprox.utils.known_rhoA2orhoB2o_distribs
    '''
    assert distrib in known.keys(), \
        'distrib=%s is not known. Please choose from : %s'%(distrib, ', '.join(known.keys()))
    assert np.all( kwargs.has_key(required) for required in known[distrib] ), \
        'distrib=%s requires kwargs : %s'%(distrib, ', '.join(known[distrib]))

    if distrib=='uniform':
        return \
            kwargs['min_rhoA2o'] + (kwargs['max_rhoA2o']-kwargs['min_rhoA2o'])*np.random.rand(Nsamp), \
            kwargs['min_rhoB2o'] + (kwargs['max_rhoB2o']-kwargs['min_rhoB2o'])*np.random.rand(Nsamp)

    elif distrib=="uniform_rho2o-fixed_eta2o":
        rho2o = kwargs['min_rho2o'] + (kwargs['max_rho2o']-kwargs['min_rho2o'])*np.random.rand(Nsamp)
        eta2o = kwargs['eta2oOVERrho2o']*rho2o
        return rho2eta2_to_rhoA2rhoB2( rho2o, eta2o )

    elif distrib=="isotropic":
        raise NotImplementedError

    elif distrib=="malmquist":
        raise NotImplementedError, 'like isotropic, but includes cuts that mimic what detection pipelines do'

    elif distrib=="background":
        return \
            sample_background( Nsamp, kwargs['nonGauss_fracA'], kwargs['min_rhoA2o'], kwargs['max_rhoA2o'], kwargs['paretoA'] ), \
            sample_background( Nsamp, kwargs['nonGauss_fracB'], kwargs['min_rhoB2o'], kwargs['max_rhoB2o'], kwargs['paretoB'] ) 

    else:
        raise ValueError, 'no sampling algorithm defined for distrib=%s. Please choose from : %s'%(distrib, ', '.join(known.keys()))

#---

def sample_background( Nsamp, nonGauss_frac, min_rho2o, max_rho2o, pareto ):
    '''
    draws samples from the expected background distribution

    p(rhoA2o) ~ (1-nonGauss_frac)*delta(rhoA2o=0) + nonGauss_frac*pareto( paretoA, min_rhoA2, max_rhoA2 )

    where pareto(a, min, max) describes a power law distribution with exponent a bounded between min and max
    '''
    samples = [1e-3]*np.sum(np.random.rand(Nsamp)>nonGauss_frac) ### get the Gaussian noise samples
    ### FIXME: 
    ###    use use 1e-3 above because things break if I supply zero exactly. We really just want the Gaussian component
    ###    to be much smaller than 1, so this should be ok...

    if len(samples) < Nsamp:
        samples += list(sample_pareto( Nsamp-len(samples), min_rho2o, max_rho2o, pareto )) ### draw nonGaussian samples from pareto

    np.random.shuffle(samples) ### randomize the order

    return samples

def sample_pareto( Nsamp, minimum, maximum, exponent ):
    '''
    samples pareto distribution
    '''
    if exponent==-1: ### special case
        return minimum*np.exp(np.log(maximum/minimum)*np.random.rand(Nsamp))

    else:
        a = exponent+1
        m = minimum**a
        return (m + np.random.rand(Nsamp)*(maximum**a - m))**(1./a)

#-------------------------------------------------

def tukey(N, alpha):
        """
        generate a tukey window
        The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
                that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
                at \alpha = 0 it becomes a Hann window. 
        """
        # Special cases
        if alpha <= 0:
                return np.ones(N) #rectangular window
        elif alpha >= 1:
                return np.hanning(N)

        # Normal case
        x = np.linspace(0, 1, N)
        w = np.ones(x.shape)

        # first condition 0 <= x < alpha/2
        first_condition = x<alpha/2
        w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))

        # second condition already taken care of

        # third condition 1 - alpha / 2 <= x <= 1
        third_condition = x>=(1 - alpha/2)
        w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2)))

        return w

def sigma_lnBSN( rho2o, params, frac=0.01 ):
    '''
    we assume the sampling error scales as a constant fraction of what was injected
    '''
    return frac*np.abs(rho2_to_lnBSN( rho2o, params ))

def sigma_singles( rhoA2o, rhoB2o, params, frac=0.01 ):
    '''
    computes the sampling error from the combination of the singles runs
    delegates to sigma_lnBSN
    '''
    return (sigma_lnBSN(rhoA2o, params, frac=frac)**2 + sigma_lnBSN(rhoB2o, params, frac=frac)**2)**0.5

def lnProb_samplingError( delta, sigma ):
    '''
    just the Gaussian distribution
    '''
    return -0.5*(delta/sigma)**2 - 0.5*np.log(2*np.pi*sigma**2)

def lnFFTProb_samplingError( atled, sigma ):
    '''
    the Fourier transform of the sampling error distrition (a Gaussian in the Fourier conjugate variable)
    NOTE: sigma is the "real-space" width which would be passed to lnProb_samplingError
    '''
    return -2*(np.pi*sigma*atled)**2

def fuckU_fft2( twodee ):
    '''
    take fft2 by hand by iterating over one of the array's indecies
    expected to be slow, but may allow me to get better numerical percision by playing tricks with the intermediate data products
    '''
    lenx, leny = twodee.shape
    ans = np.empty((lenx, leny), dtype=complex)
    for ind in xrange(lenx):
        ans[ind,:] = np.fft.fft( twodee[ind,:] )
    for ind in xrange(leny):
        ans[:,ind] = np.fft.fft( ans[:,ind] )
    return ans

def fuckU_ifft2( twodee ):
    '''
    take ifft2 by hand by iterating over array's indecies in the opposite order as fuckU_fft2
    expected to be slow, but may allow me to get better numerical precision by playing tricks with the intermediate data products
    '''
    lenx, leny = twodee.shape
    ans = np.empty((lenx, leny), dtype=complex)
    for ind in xrange(leny):
        ans[:,ind] = np.fft.ifft( twodee[:,ind] )
    for ind in xrange(lenx):
        ans[ind,:] = np.fft.ifft( ans[ind,:] )
    return ans

def fuckU_convolve_samplingErrors( lnBSN, lnBCI, rhoA2o, rhoB2o, params, frac=0.01, tukey_alpha=0.1, f_tol=1e-10 ):
    '''
    we should take the fft and ifft of only one dimension at a time and do this by hand.
    this should allow us to determine whether fft2, ifft2 are broken or the mistake is mine

    I expect it to be slower due to the python iteration requied, 
    but hopefully we can get it to return real values instead of complex
    (slow, but correct)

    we use a Tukey window for both lnBSN and lnBCI separately

    NOTE: 
        this assumes that lnBSN, lnBCI, lnProb are all numpy.ndarray objects with the same shape and the structure expected for something like matplotlib.pyplot.contours
        rho2Ao, rho2Bo, params, frac are used to determine the expected sampling errors
        tukey_alpha is used to constructe a windowing function used in the FFT. We do not use a windowing function in the iFFT because it probably isn't necessary (the convolution kernels should kill most of that stuff)
    '''
    ### compute the probability instead of taking it in as an argument
    N = np.prod(lnBSN.shape)
    lnProb = lnProb_lnBSNlnBCI_given_rhoA2orhoB2o(
        lnBSN.flatten(),
        lnBCI.flatten(),
        rhoA2o*np.ones(N, dtype=float),
        rhoB2o*np.ones(N, dtype=float),
        params,
        f_tol=f_tol
    ).reshape(lnBSN.shape)

    ### FFT lnProb    
    max_lnProb = np.max(lnProb) ### do everything releative to the maximum value to improve numerical precision
    lnProb -= max_lnProb

    # windowing function
    size_lnBSN, size_lnBCI = lnProb.shape ### get the dimensionality of lnProb
    win = np.outer( tukey(size_lnBSN, tukey_alpha), tukey(size_lnBCI, tukey_alpha) )

    # determine conjugate variable values corresponding to FFT array
    frq_lnBSN, frq_lnBCI = np.meshgrid(
        np.fft.fftfreq(size_lnBSN, d=lnBSN[0,1]-lnBSN[0,0]),
        np.fft.fftfreq(size_lnBCI, d=lnBCI[1,0]-lnBCI[0,0]),
    )

    # actual FFT
    fftProb = fuckU_fft2( np.exp(lnProb)*win )

    ### multiply by Fourier conjugates of the sampling error distributions
    lnFFT_lnBSNerr = lnFFTProb_samplingError(frq_lnBSN+frq_lnBCI, sigma_lnBSN(rhoA2rhoB2_to_rho2eta2(rhoA2o, rhoB2o)[0], params, frac=frac))
    max_lnFFT_lnBSNerr = np.max(lnFFT_lnBSNerr) ### remove this for numerical precision
    lnFFT_lnBSNerr -= max_lnFFT_lnBSNerr

    lnFFT_lnBCIerr = lnFFTProb_samplingError(frq_lnBCI, sigma_singles(rhoA2o, rhoB2o, params, frac=frac))
    max_lnFFT_lnBCIerr = np.max(lnFFT_lnBCIerr) ### remove for numerical precision
    lnFFT_lnBCIerr -= max_lnFFT_lnBCIerr

    # multiply by fourier transform of distributions
    fftProb *= np.exp(lnFFT_lnBSNerr + lnFFT_lnBCIerr)

    ### iFFT the result
    result = fuckU_ifft2( np.fft.ifftshift(np.fft.fftshift(fftProb)*win) ) ### tukey this again to kill aliasing

    ### take log and add back the factors that I kept out for precision
    return np.log(result) + max_lnProb + max_lnFFT_lnBSNerr + max_lnFFT_lnBCIerr

def convolve_samplingErrors( lnBSN, lnBCI, rhoA2o, rhoB2o, params, frac=0.01, tukey_alpha=0.1, f_tol=1e-10 ):
    '''
    marginalize over sampling errors numerically via spectral convolution
    we use a Tukey window for both lnBSN and lnBCI separately

    NOTE: 
        this assumes that lnBSN, lnBCI, lnProb are all numpy.ndarray objects with the same shape and the structure expected for something like matplotlib.pyplot.contours
        rho2Ao, rho2Bo, params, frac are used to determine the expected sampling errors
        tukey_alpha is used to constructe a windowing function used in the FFT. We do not use a windowing function in the iFFT because it probably isn't necessary (the convolution kernels should kill most of that stuff)
    '''
    ### compute the probability instead of taking it in as an argument
    N = np.prod(lnBSN.shape)
    lnProb = lnProb_lnBSNlnBCI_given_rhoA2orhoB2o(
        lnBSN.flatten(), 
        lnBCI.flatten(), 
        rhoA2o*np.ones(N, dtype=float), 
        rhoB2o*np.ones(N, dtype=float), 
        params, 
        f_tol=f_tol
    ).reshape(lnBSN.shape)
    
    ### FFT lnProb    
    max_lnProb = np.max(lnProb) ### do everything releative to the maximum value to improve numerical precision
    lnProb -= max_lnProb

    # windowing function
    size_lnBSN, size_lnBCI = lnProb.shape ### get the dimensionality of lnProb
    win = np.outer( tukey(size_lnBSN, tukey_alpha), tukey(size_lnBCI, tukey_alpha) )

    # determine conjugate variable values corresponding to FFT array
    frq_lnBSN, frq_lnBCI = np.meshgrid( 
        np.fft.fftfreq(size_lnBSN, d=lnBSN[0,1]-lnBSN[0,0]), 
        np.fft.fftfreq(size_lnBCI, d=lnBCI[1,0]-lnBCI[0,0]),
    )

    # actual FFT
    fftProb = np.fft.fft2( np.exp(lnProb)*win )
#    fftProb = np.fft.fft2( np.exp(lnProb) ) ### windowing also appears to affect whether the final answer is real... what?

    ### multiply by Fourier conjugates of the sampling error distributions
    lnFFT_lnBSNerr = lnFFTProb_samplingError(frq_lnBSN+frq_lnBCI, sigma_lnBSN(rhoA2rhoB2_to_rho2eta2(rhoA2o, rhoB2o)[0], params, frac=frac))
    max_lnFFT_lnBSNerr = np.max(lnFFT_lnBSNerr) ### remove this for numerical precision
    lnFFT_lnBSNerr -= max_lnFFT_lnBSNerr

    lnFFT_lnBCIerr = lnFFTProb_samplingError(frq_lnBCI, sigma_singles(rhoA2o, rhoB2o, params, frac=frac))
    max_lnFFT_lnBCIerr = np.max(lnFFT_lnBCIerr) ### remove for numerical precision
    lnFFT_lnBCIerr -= max_lnFFT_lnBCIerr

    # multiply by fourier transform of distributions
    ### FIXME it seems like this is returning complex values with large imaginary components
    fftProb *= np.exp(lnFFT_lnBSNerr + lnFFT_lnBCIerr)  ### somehow, the lnFFT_lnBSNerr term messes up ifft2...
#    fftProb *= np.exp(lnFFT_lnBCIerr)

    ### iFFT the result
#    result = np.fft.ifft2( fftProb )
    result = fuckU_ifft2( np.fft.ifftshift(np.fft.fftshift(fftProb)*win) ) ### tukey this again to kill aliasing

#    truth = np.abs(result.real)>1e-15
#    print result[truth]
#    print result.imag[truth]/result.real[truth]

    ### take log and add back the factors that I kept out for precision
    return np.log(result) + max_lnProb + max_lnFFT_lnBSNerr + max_lnFFT_lnBCIerr

def importanceSample_samplingErrors( lnBSN, lnBCI, rhoA2o, rhoB2o, params, frac=0.01, f_tol=1e-10, Nsamp=100 ):
    '''
    marginalize over sampling errors via importance sampling

    we draw Nsamp samples at each point of the grid represented by (lnBSN, lnBCI)
    NOTE:
        this assumes (lnBSN, lnBCI) are each 1D numpy.ndarray objects
    '''
    # get samples from normal distributions
    N = len(lnBSN)
    delta1 = np.random.randn(N, Nsamp)*sigma_lnBSN(rhoA2rhoB2_to_rho2eta2(rhoA2o, rhoB2o)[0], params, frac=frac)
    delta2 = np.random.randn(N, Nsamp)*sigma_singles(rhoA2o, rhoB2o, params, frac=frac)

    # shift the sample points by the samples of sampling errors
    lnBSN = np.outer(lnBSN, np.ones(Nsamp, dtype=float)) + delta1
    lnBCI = np.outer(lnBCI, np.ones(Nsamp, dtype=float)) + delta1 + delta2

    # compute lnProb for all these points, reshape, and marginalize
    n = np.prod(lnBSN.shape)
    rhoA2o *= np.ones(n, dtype=float)
    rhoB2o *= np.ones(n, dtype=float)
    return sumLogs(lnProb_lnBSNlnBCI_given_rhoA2orhoB2o(lnBSN.flatten(), lnBCI.flatten(), rhoA2o, rhoB2o, params, f_tol=f_tol).reshape(N, Nsamp), axis=1) - np.log(Nsamp)

def directIntegrate_samplingErrors( lnBSN, lnBCI, rhoA2o, rhoB2o, params, frac=0.01, f_tol=1e-10, Nint=10 ):
    '''
    we expect lnBSN, lnBCI to be 1D np.array objects

    Nint is the number of points used to perform the numeric integration. Nint is used on half the plane, so each integral will acutally use 2*Nint-1 samples (due to repeated zero point)

    NOTE: my integration strategy truncates the gaussian integral over sampling errors at the last points (+/- inf are dropped) so that I can handle the weighting by un-even sampling. This should be a small effect as long as Nint is reasonably large.
    I do weight the integral by the prior weight accumulated during this approximate integration. This should help mitigate errors, but a large-ish Nint is still likely to be required.

    WARNING: this could be made more memory efficient by restructuring some of the loops. For instance, if we're really only going to sum up the numeric marginalizations one point at a time in a nested loop, there's no reason to compute all of lnProb for all points we'll ever need at the beginning. We can just compute exactly what we need within that loop and go from there. Should prevent the need for really big arrays.
    Of course, if I can figure out how to replace the nested loops with array manipulations, then everything should just be faster.
    '''
    N = len(lnBSN)

    ### set up the points used to evaluate the integrals
    samples = erfinv(np.linspace(0,1,Nint+1))[:-1]
    samples = np.concatenate([-samples[::-1][:-1], samples])
    L = len(samples)
    ### compute sampling error standard deviations
    sigma1 = sigma_lnBSN(rhoA2rhoB2_to_rho2eta2(rhoA2o, rhoB2o)[0], params, frac=frac)
    sigma2 = sigma_singles(rhoA2o, rhoB2o, params, frac=frac)

    ### compute a grid of points for sampling errors
    delta1, delta2 = np.meshgrid(
        samples*sigma1, 
        samples*sigma2,
    )
    delta1 = np.outer( np.ones(N, dtype=float), delta1.flatten() ).flatten()
    delta2 = np.outer( np.ones(N, dtype=float), delta2.flatten() ).flatten()
    M = L**2

    ### now, compute probability for all points
    lnProb = lnProb_lnBSNlnBCI_given_rhoA2orhoB2o(
        np.outer(lnBSN, np.ones(M, dtype=float)).flatten() + delta1, 
        np.outer(lnBCI, np.ones(M, dtype=float)).flatten() + delta1 + delta2, 
        rhoA2o*np.ones(M*N, dtype=float), 
        rhoB2o*np.ones(M*N, dtype=float), 
        params, 
        f_tol=f_tol,
    )

    ### weight by sampling errors 
    sampErr1 = lnProb_samplingError( delta1, sigma1 )
    sampErr2 = lnProb_samplingError( delta2, sigma2 )
    lnProb += sampErr2 + sampErr2

    ### sum to compute integral
    lnProb = lnProb.reshape((N,L,L))

    # go over bci integral first
    answer = np.empty(N, dtype=float)
    delta1 = delta1.reshape((N,L,L))
    delta2 = delta2.reshape((N,L,L))
    sampErr1 = sampErr1.reshape((N,L,L))
    sampErr2 = sampErr2.reshape((N,L,L))
    for i in xrange(N):
        ans = np.empty(L, dtype=float)
        for j in xrange(L):
            weight = np.log(delta2[i,1:,j]-delta2[i,:-1,j]) ### weight by dDelta

            # actual integral
            wer = sumLogs(np.array([lnProb[i,:-1,j], lnProb[i,1:,j]]), axis=0) - np.log(2) ### average neighboring points
            ans[j] = sumLogs(wer+weight)

            # the prior weight
            wer = sumLogs(np.array([sampErr2[i,:-1,j], sampErr2[i,1:,j]]), axis=0) - np.log(2)
            ans[j] -= sumLogs(wer+weight)

            try:
                assert np.all(ans[j]==ans[j])
            except AssertionError as e:
                blah = sumLogs(np.array([lnProb[i,:-1,j], lnProb[i,1:,j]]), axis=0)
                print blah
                raise e

        weight = np.log(delta1[i,0,1:]-delta1[i,0,:-1]) ### weight by dDelta

        # the actual integral
        ans = sumLogs(np.array([ans[:-1], ans[1:]]), axis=0) - np.log(2) ### average neighboring points
        answer[i] = sumLogs(ans-weight)

        # the prior weight
        ans = sumLogs(np.array([sampErr1[i,0,1:], sampErr1[i,0,:-1]]), axis=0) - np.log(2) ### average neighboring points
        answer[i] -= sumLogs(ans-weight)

    return answer
