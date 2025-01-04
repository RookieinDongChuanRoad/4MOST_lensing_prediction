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

#* prepare interpolate grid for gnfw
bgrid_min = 0.2
bgrid_max = 2.8
Nb = 27

Rgrid_min = 0.001
Rgrid_max = 100.
Nr = 100

beta_grid = np.linspace(bgrid_min, bgrid_max, Nb)
R_grid = np.logspace(np.log10(Rgrid_min), np.log10(Rgrid_max), Nr)

axes = {0: splrep(beta_grid, np.arange(Nb)), 1: splrep(R_grid, np.arange(Nr))}

#* found the gird file and interpolate
with h5py.File('/Users/liurongfu/Desktop/4MOST_lensing_prediction/codes/SL_simulation/profiles/gnfw_grids.hdf5', 'r') as grid_file:
    rgrid = grid_file['R_grid'][()]
    gammagrid = grid_file['beta_grid'][()]
    R,B= np.meshgrid(rgrid,gammagrid)
    sigmagrid = grid_file['Sigma_grid'][()]
    Mgrid = grid_file['M2d_grid'][()]
    Sigma_interp = ndinterp.ndInterp(axes, sigmagrid*R, order=3)
    M_interp = ndinterp.ndInterp(axes, Mgrid*R**(B-3.), order=3)


# =============================================================================
def Sigma(r, rs, gamma_DM):
    """The surface density of the gnfw profile
        Note here we do not consider the factor of rhos, as we can mulitply it outside the function

    Parameters
    ----------
    r : float or array_like
        2D radius with respect to the center
    rs : float or array_like
        2D scale radius of gnfw profile
    gamma_DM : float or array_like
        the inner denstiy slope of gnfw
    """

    r_array = np.atleast_1d(r)
    out = np.zeros_like(r_array)
    for i in range(len(r_array)):
        out[i]  = 2*rs*(r_array[i]/rs)**(1-gamma_DM)*quad(lambda x: np.sin(x)*(np.sin(x)+ r_array[i]/rs)**(gamma_DM-3),0,np.pi/2)[0]
    
    return out

def M_enclosed_2D(r, rs, gamma_DM):
    """The enclosed mass of the gnfw

    Parameters
    ----------
    r : float or array_like
        2D radius with respect to the center
    rs : float or array_like
        2D scale radius of gnfw profile
    gamma_DM : float or array_like
        the inner denstiy slope of gnfw
    """
    r_array = np.atleast_1d(r)
    out = np.zeros_like(r_array)
    for i in range(len(r_array)):
        out[i] = 2*np.pi*quad(lambda x: x*Sigma(x,rs,gamma_DM),0,r_array[i])[0]
    return out

def Sigma_fast(r,rs,gamma_DM):
    """Faster version of the Sigma function

    Parameters
    ----------
    r : float or array_like
        2D radius with respect to the center
    rs : float or array_like
        2D scale radius of gnfw profile
    gamma_DM : float or array_like
        the inner denstiy slope of gnfw
    """
    r = np.atleast_1d(r)
    rs = np.atleast_1d(rs)
    gamma_DM = np.atleast_1d(gamma_DM)
    length = max(len(r),len(rs),len(gamma_DM))
    sample = np.array([gamma_DM*np.ones(length),r/rs*np.ones(length)]).T
    Sigma_here = Sigma_interp.eval(sample)/(r/rs)/rs**2
    return Sigma_here*rs**3

def M_enclosed_2D_fast(r,rs,gamma_DM):
    """Faster version of the M2d function

    Parameters
    ----------
    r : float or array_like
        2D radius with respect to the center
    rs : float or array_like
        2D scale radius of gnfw profile
    gamma_DM : float or array_like
        the inner denstiy slope of gnfw
    """
    r = np.atleast_1d(r)
    rs = np.atleast_1d(rs)
    gamma_DM = np.atleast_1d(gamma_DM)
    length = max(len(r),len(rs),len(gamma_DM))
    sample = np.array([gamma_DM*np.ones(length),r/rs*np.ones(length)]).T
    M2d_here = M_interp.eval(sample)*(r/rs)**(3-gamma_DM)
    return M2d_here*rs**3

def Sigma_c_func(z_lens,z_source):
    """Critical surface density of the gnfw profile

    Parameters
    ----------
    z_lens : float or array_like
        redshift of the lens
    z_source : float or array_like
        redshift of the source
    """
    D_lens = cosmo_astropy.angular_diameter_distance(z_lens)
    D_source = cosmo_astropy.angular_diameter_distance(z_source)
    D_lens_source = cosmo_astropy.angular_diameter_distance_z1z2(z_lens, z_source)
    Sigma_c = c**2/(4*np.pi*G)*D_source/(D_lens*D_lens_source)
    Sigma_c = Sigma_c.to('Msun/kpc^2').value
    return Sigma_c

def kappa(r,rs,gamma_DM,z_lens,z_source,rhos):
    """Dimensionless surface density of the gnfw profile

    Parameters
    ----------
    r : float
        2D radius with respect to the center
    rs : float
        2D scale radius of gnfw profile
    gamma_DM : float
        the inner denstiy slope of gnfw
    z_lens : float
        redshift of the lens
    z_source : float
        redshift of the source
    rhos : float
        the inner density of gnfw
    """ 
    Sigma_c = Sigma_c_func(z_lens,z_source)
    return rhos*Sigma_fast(r,rs,gamma_DM)/Sigma_c

def alpha(r, rs, gamma_DM, z_lens, z_source, rhos):
    """Defelction angle of the gnfw profile

    Parameters
    ----------
    r : float
        2D radius with respect to the center
    rs : float
        2D scale radius of gnfw profile
    gamma_DM : float
        the inner denstiy slope of gnfw
    z_lens : float
        redshift of the lens
    z_source : float
        redshift of the source
    rhos : float
        the inner density of gnfw
    """
    Sigma_c = Sigma_c_func(z_lens, z_source)
    return rhos*M_enclosed_2D_fast(r,rs,gamma_DM)/Sigma_c/r/np.pi