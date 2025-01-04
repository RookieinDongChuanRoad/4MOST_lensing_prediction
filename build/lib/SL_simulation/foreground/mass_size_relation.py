import numpy as np
from scipy.stats import norm



def mu_logRe(logM):
    '''
    Median of effective radius at a given stellar mass
    Parameters
    ----------
    logM : float or array_like
        log10(M*/Msun)
    '''
    mu_R = 0.99 
    beta_R = 0.61
    return mu_R + beta_R*(logM - 11.4)

def PDF(logRe,logM):
    '''
    PDF of effective radius at a given stellar mass
    Parameters
    ----------
    logRe : float or array_like
        log10(Re/kpc)
    logM : float or array_like
        log10(M*/Msun)
    '''
    mu_R = mu_logRe(logM)
    sigma_R = 0.20
    return norm.pdf(logRe,loc=mu_R,scale=sigma_R)

def CDF(logRe,logM):
    '''
    CDF of effective radius at a given stellar mass
    Parameters
    ----------
    logRe : float or array_like
        log10(Re/kpc)
    logM : float or array_like
        log10(M*/Msun)
    '''
    mu_R = mu_logRe(logM)
    sigma_R = 0.20
    return norm.cdf(logRe,loc=mu_R,scale=sigma_R)

def CDF_inv(y,logM):
    '''
    Inverse CDF of effective radius at a given stellar mass
    Parameters
    ----------
    y : float or array_like
        CDF value
    logM : float or array_like
        log10(M*/Msun)
    '''
    mu_R = mu_logRe(logM)
    sigma_R = 0.20
    return norm.ppf(y,loc=mu_R,scale=sigma_R)

