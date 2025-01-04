import numpy as np
from scipy.interpolate import splrep, splev
from scipy.integrate import quad
from scipy.stats import norm
from colossus.cosmology import cosmology
from colossus.lss import mass_function
import astropy.cosmology

cosmo_colossus = cosmology.setCosmology('planck13')
cosmo_astropy = cosmo_colossus.toAstropy()

def mu_Mstar_Mh(logMh):
    '''
    The mean value of Mstar given Mh, from Shuntov et al. 2022
    Parameters
    ----------
    logMh : float
        log10(Mh/Msun)
    '''
    logM0 = 12.629
    logM1 = 10.855
    beta = 0.487
    delta = 0.935
    gamma = 1.939

    #* Shuntov give this relation inversly, i.e. Mh(Mstar), we need to invert it
    logmstar_interp = np.linspace(10,11.8,1000)
    logmh_interp = logM0 + beta*(logmstar_interp- logM1) + (10**(logmstar_interp - logM1))**delta / (1 + (10**(logmstar_interp - logM1))**(-gamma)) - 0.5
    #spline inversly
    spline = splrep(logmh_interp,logmstar_interp)
    return splev(logMh,spline)

def P_Mstar_Mh(logMstar,logMh):
    '''
    The probability of Mstar given Mh, from Shuntov et al. 2022
    Parameters
    ----------
    logMstar : float
        log10(Mstar/Msun)
    logMh : float
        log10(Mh/Msun)
    '''
    mu_Mstar = mu_Mstar_Mh(logMh)
    sigma_logMstar = 0.268
    return norm.pdf(logMstar,loc=mu_Mstar,scale=sigma_logMstar)

def P_Mh(logMh):
    '''
    The marginalized probability of Mh, i.e. the normalized halo mass function
    (We fix the redshift to be z=0.1)
    Parameters
    ----------
    logMh : float or array_like
        log10(Mh/Msun)
    '''
    logMh = np.atleast_1d(logMh)
    return mass_function.massFunction(10**logMh,0.1,q_in = 'M', mdef = 'vir',q_out='dndlnM',model = 'despali16')

logMh_array = np.linspace(11.5,18,1000)
logMstar_array = np.linspace(10,11.8,1000)
mean_logMh_ar = np.zeros_like(logMstar_array)
std_logMh_ar = np.zeros_like(logMstar_array)
for i in range(len(logMstar_array)):
    distribute = P_Mstar_Mh(logMstar_array[i],logMh_array)*P_Mh(logMh_array)
    normalized_distribute = distribute/np.sum(distribute)
    mean_logMh_ar[i] = np.sum(normalized_distribute*logMh_array)
    std_logMh_ar[i] = np.sqrt(np.sum(normalized_distribute*(logMh_array-mean_logMh_ar[i])**2))

spline_mh_mstar = splrep(logMstar_array,mean_logMh_ar)
spline_std_mh_mstar = splrep(logMstar_array,std_logMh_ar)

def mu_Mh_Mstar(logMstar):
    '''
    The mean value of Mh given Mstar
    Parameters
    ----------
    logMstar : float
        log10(Mstar/Msun)
    '''
    return splev(logMstar,spline_mh_mstar)

def sigma_Mh_Mstar(logMstar):
    '''
    The standard deviation of Mh given Mstar
    Parameters
    ----------
    logMstar : float
        log10(Mstar/Msun)
    '''
    return splev(logMstar,spline_std_mh_mstar)

def PDF(logMstar,logMh):
    '''
    The probability distribution function MH given Mstar
    Parameters
    ----------
    logMstar : float or array_like
        log10(Mstar/Msun)
    logMh : float or array_like
        log10(Mh/Msun)
    '''
    mu_Mh = mu_Mh_Mstar(logMstar)
    sigma_Mh = sigma_Mh_Mstar(logMstar)
    return norm.pdf(logMh,loc=mu_Mh,scale=sigma_Mh)

def CDF(logMstar,logMh):
    '''
    The cumulative distribution function MH given Mstar
    Parameters
    ----------
    logMstar : float or array_like
        log10(Mstar/Msun)
    logMh : float or array_like
        log10(Mh/Msun)
    '''
    mu_Mh = mu_Mh_Mstar(logMstar)
    sigma_Mh = sigma_Mh_Mstar(logMstar)
    return norm.cdf(logMh,loc=mu_Mh,scale=sigma_Mh)

def CDF_inv(F,logMstar):
    '''
    The inverse function of CDF
    Parameters
    ----------
    F : float or array_like
        The cumulative distribution function
    logMstar : float or array_like
        log10(Mstar/Msun)
    '''
    mu_Mh = mu_Mh_Mstar(logMstar)
    sigma_Mh = sigma_Mh_Mstar(logMstar)
    return norm.ppf(F,loc=mu_Mh,scale=sigma_Mh)