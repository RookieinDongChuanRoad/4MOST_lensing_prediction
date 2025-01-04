import numpy as np
import h5py
from colossus.halo import mass_so
from scipy.integrate import quad
from scipy.special import gamma,gammainc
from scipy.interpolate import splrep,splev
from SL_simulation import ndinterp
import os
from colossus.cosmology import cosmology
import astropy.cosmology
from astropy.constants import G,c

cosmo_colossus = cosmology.setCosmology('planck13')
cosmo_astropy = cosmo_colossus.toAstropy()
c = c.cgs
G = G.cgs

delta_Vir = mass_so.deltaVir

def b_n(n):
    return 2 * n - 1/3 + 4/(405 * n) + 46/(25515 * n**2) + 131/(1148175 * n**3) - 2194697/(30690717750 * n**4)

def Sigma(r,Re,n):
    """Surface density of Sersic profile at radius r

    Parameters
    ----------
    r : float or array_like
        2D radius with respect to the center
    Re : float 
        Effective radius of the Sersic profile
    n : float
        Sersic index
    """
    r_array = np.atleast_1d(r)
    bn = b_n(n)
    out = np.zeros_like(r_array)
    I0 = 1/(Re**2 * 2 * np.pi * n * gamma(2 * n) / (bn**(2 * n)))
    for i in range(len(r_array)):
        out[i] = I0 * np.exp(-bn * (r_array[i]/Re)**(1/n))
    return out

def M_enclosed_2D(r,Re,n):
    """The fraction of Enclosed mass of Sersic profile at radius r
        Multiply the total mass to get the enclosed mass
    Parameters
    ----------
    r : float or array_like
        2D radius with respect to the center
    Re : float 
        Effective radius of the Sersic profile
    n : float
        Sersic index
    """
    r = np.atleast_1d(r)
    bn = b_n(n)
    out = np.zeros_like(r)
    for i in range(len(r)):
        out[i] = gammainc(2 * n, bn * (r[i]/Re)**(1/n))
    return out

def Sigma_c_func(z_lens,z_source):
    """Critical surface density of the Sersic profile

    Parameters
    ----------
    z_lens : float
        Redshift of the lens
    z_source : float
        Redshift of the source
    """
    D_lens = cosmo_astropy.angular_diameter_distance(z_lens)
    D_source = cosmo_astropy.angular_diameter_distance(z_source)
    D_lens_source = cosmo_astropy.angular_diameter_distance_z1z2(z_lens, z_source)
    Sigma_c = c**2/(4 * np.pi * G) * D_source/(D_lens * D_lens_source)
    Sigma_c = Sigma_c.to('Msun/kpc^2').value
    return Sigma_c

def kappa(r,Re,n,z_lens,z_source,Mstar):
    """Dimensionless surface density of the Sersic profile

    Parameters
    ----------
    r : float or array_like
        2D radius with respect to the center
    Re : float 
        Effective radius of the Sersic profile
    n : floatß
        Sersic index
    z_lens : float
        Redshift of the lens
    z_source : float
        Redshift of the source
    Mstar : float
        Total stellar mass of the Sersic profile
    """
    Sigma_c = Sigma_c_func(z_lens,z_source)
    return Mstar * Sigma(r,Re,n) / Sigma_c

def alpha(r,Re,n,z_lens,z_source,Mstar):
    """Deflection angle of the Sersic profile

    Parameters
    ----------
    r : float or array_like
        2D radius with respect to the center
    Re : float 
        Effective radius of the Sersic profile
    n : float
        Sersic index
    z_lens : float
        Redshift of the lens
    z_source : float
        Redshift of the source
    Mstar : float
        Total stellar mass of the Sersic profile
    """
    Sigma_c = Sigma_c_func(z_lens, z_source)
    return Mstar * M_enclosed_2D(r,Re,n) / Sigma_c / r / np.pi
