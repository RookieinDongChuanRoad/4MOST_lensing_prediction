import numpy as np
from scipy.interpolate import splrep,splev
from scipy.integrate import quad
from scipy.stats import norm



def PDF(logM):
    """Probability density function of stellar mass function
    From Driver et al. 2022
    Parameters
    ----------
    logM : float or array_like
        log10 stellar mass
    """
    logM_star = 10.954
    logM_star_err = 0.028
    logphi = -2.994
    logphi_err = 0.025
    alpha = -0.524
    alpha_err = 0.037
    
    return 10**(logphi) * 10**((logM - logM_star) * (1 + alpha)) * np.exp(-10**(logM - logM_star ))

logM_min = 11
logM_max = 12.5
normalization = 1/quad(PDF, logM_min, logM_max)[0]
x_interpolate = np.linspace(logM_min, logM_max, 1000)
y_interpolate = np.array([quad(PDF, logM_min, x)[0]*normalization for x in x_interpolate])
cdf_spline = splrep(x_interpolate, y_interpolate)
unique_indices = np.unique(y_interpolate, return_index=True)[1]
x_interpolate_unique = x_interpolate[unique_indices]
y_interpolate_unique = y_interpolate[unique_indices]
cdf_inv_spline = splrep(y_interpolate_unique, x_interpolate_unique)

def CDF(logM):
    """Cumulative distribution function of stellar mass function
    From Driver et al. 2022
    Parameters
    ----------
    logM : float or array_like
        log10 stellar mass
    """
    return splev(logM, cdf_spline)

def CDF_inv(y):
    """Inverse of cumulative distribution function of stellar mass function
    From Driver et al. 2022
    Parameters
    ----------
    y : float or array_like
        Cumulative probability
    """
    return splev(y, cdf_inv_spline)
