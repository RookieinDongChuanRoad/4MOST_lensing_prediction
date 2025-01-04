import numpy as np
from scipy.interpolate import splrep,splev
from scipy.integrate import quad

def PDF(q):
    """Probability density function of ellipticity
    From Sonnenfeld 2023
    Parameters
    ----------
    q : float or array_like
        Ellipticity
    """
    alpha = 6.28
    beta = 2.05
    
    return q**(6.28)*(1-q)**(2.05)

#* interpolate to get CDF and inverse CDF
normalization = 1/quad(PDF, 0, 1)[0]
x_interpolate = np.linspace(0, 1, 1000)
y_interpolate = np.array([quad(PDF, 0, x)[0]*normalization for x in x_interpolate])
# make sure y_interpolate is unique
unique_indices = np.unique(y_interpolate, return_index=True)[1]
x_interpolate_unique = x_interpolate[unique_indices]
y_interpolate_unique = y_interpolate[unique_indices]
# get the spline
cdf_spline = splrep(x_interpolate_unique, y_interpolate_unique)
cdf_inv_spline = splrep(y_interpolate_unique, x_interpolate_unique)

def CDF(q):
    """Cumulative distribution function of ellipticity
    From Sonnenfeld 2023
    Parameters
    ----------
    q : float or array_like
        Ellipticity
    """
    return splev(q, cdf_spline)

def CDF_inv(y):
    """Inverse of cumulative distribution function of ellipticity
    From Sonnenfeld 2023
    Parameters
    ----------
    y : float or array_like
        Cumulative probability
    """
    return splev(y, cdf_inv_spline)