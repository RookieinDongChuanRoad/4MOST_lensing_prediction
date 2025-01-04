import numpy as np
from scipy.interpolate import splrep, splev
from scipy.stats import norm
from colossus.cosmology import cosmology    
from colossus.halo import concentration
import astropy.cosmology

cosmo_colossus = cosmology.setCosmology('planck13')
cosmo_astropy = cosmo_colossus.toAstropy()

def mu_logc(logMh):
    """median log concentration as a function of halo mass

    Parameters
    ----------
    logMh : float or array_like
        log10(Mh/Msun)
    """
    return np.log10(concentration.concentration(10**logMh, 'vir',0.1,model = 'klypin16_m'))

def CDF_inv(logMh, F):
    """inverse of the cumulative distribution function of log concentration

    Parameters
    ----------
    logMh : float or array_like
        log10(Mh/Msun)
    F : float or array_like
        cumulative distribution function
    """
    mu = mu_logc(logMh)
    sigma = 0.1
    return norm.ppf(F, loc=mu, scale=sigma)