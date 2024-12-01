import numpy as np
import h5py
import foreground_model as fg
from colossus.halo import mass_so
from scipy.integrate import quad
from scipy.special import gamma,gammainc
from scipy.interpolate import splrep,splev
import ndinterp
from numba import jit
import os

from astropy.constants import G,c
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



class gnfw:
    def __init__(self):

        #* some cosmological parameters
        self.cosmo_astropy = fg.cosmo_astropy
        self.deltaVir = mass_so.deltaVir

        #* interpolate the Sigma and M2d grid to accelerate the calculation
        grid_file = h5py.File('/root/4MOST_lensing_prediction/data/gnfw_grids.hdf5','r')
        self.rgrid = grid_file['R_grid'][()]
        self.gammagrid = grid_file['beta_grid'][()]
        self.R,self.B= np.meshgrid(self.rgrid,self.gammagrid)
        self.sigmagrid = grid_file['Sigma_grid'][()]
        self.Mgrid = grid_file['M2d_grid'][()]
        grid_file.close()
        self.Sigma_interp = ndinterp.ndInterp(axes, self.sigmagrid*self.R, order=3)
        self.M_interp = ndinterp.ndInterp(axes, self.Mgrid*self.R**(self.B-3.), order=3)

    #* the surface density of the gnfw profile
    def Sigma(self,r, rs, gamma_DM):
        #* Here we do not consider the factor of rhos, as we can mulitply it outside the function
        r_array = np.atleast_1d(r)
        out = np.zeros_like(r_array) 
        for i in range(len(r_array)):
            out[i]  = 2*rs*(r_array[i]/rs)**(1-gamma_DM)*quad(lambda x: np.sin(x)*(np.sin(x)+ r_array[i]/rs)**(gamma_DM-3),0,np.pi/2)[0]
                    
        return out
    #* the enclosed mass of the gnfw profile    
    def M_enclosed_2D(self,r, rs, gamma_DM):
        r_array = np.atleast_1d(r)
        out = np.zeros_like(r_array)
        for i in range(len(r_array)):
            # print(quad(lambda x: x*self.Sigma(x,rs,gamma_DM),0,r_array[i]))
            out[i] = 2*np.pi*quad(lambda x: x*self.Sigma(x,rs,gamma_DM),0,r_array[i])[0]
        return out
    
    #* faster version of the Sigma function
    def Sigma_fast(self,r,rs,gamma_DM):
        r = np.atleast_1d(r)
        rs = np.atleast_1d(rs)
        gamma_DM = np.atleast_1d(gamma_DM)
        length = max(len(r),len(rs),len(gamma_DM))
        sample = np.array([gamma_DM*np.ones(length),r/rs*np.ones(length)]).T
        Sigma_here = self.Sigma_interp.eval(sample)/(r/rs)/rs**2
        return Sigma_here*rs**3
    
    #* faster version of the M2d function
    def M_enclosed_2D_fast(self,r,rs,gamma_DM):
        r = np.atleast_1d(r)
        rs = np.atleast_1d(rs)
        gamma_DM = np.atleast_1d(gamma_DM)
        length = max(len(r),len(rs),len(gamma_DM))
        sample = np.array([gamma_DM*np.ones(length),r/rs*np.ones(length)]).T
        M2d_here = self.M_interp.eval(sample)*(r/rs)**(3-gamma_DM)
        return M2d_here*rs**3
    
    #* critical surface density of the gnfw profile
    def Sigma_c(self,z_lens,z_source):
        D_lens = self.cosmo_astropy.angular_diameter_distance(z_lens)
        D_source = self.cosmo_astropy.angular_diameter_distance(z_source)
        D_lens_source = self.cosmo_astropy.angular_diameter_distance_z1z2(z_lens, z_source)
        Sigma_c = c**2/(4*np.pi*G)*D_source/(D_lens*D_lens_source)
        Sigma_c = Sigma_c.to('Msun/kpc^2').value
        return Sigma_c

    #* dimensionless surface density of the gnfw profile
    def kappa(self,r,rs,gamma_DM,z_lens,z_source,rhos):
        Sigma_c = self.Sigma_c(z_lens,z_source)
        return rhos*self.Sigma_fast(r,rs,gamma_DM)/Sigma_c

    #* defelction angle of the gnfw profile
    def alpha(self, r, rs, gamma_DM, z_lens, z_source, rhos):
        #* assume the lens lays in the center of the coordinate
        Sigma_c = self.Sigma_c(z_lens, z_source)
        return rhos*self.M_enclosed_2D_fast(r,rs,gamma_DM)/Sigma_c/r/np.pi



    
class deV:
    def __init__(self):
        self.cosmo_astropy = fg.cosmo_astropy
        self.deltaVir = mass_so.deltaVir
        self.n = 4 #* n = 4 for devaucouleurs profile
        self.bn = 7.669

    def Sigma(self,r,Re):
        r_array = np.atleast_1d(r)
        out = np.zeros_like(r)
        I0 = 1/(Re**2*2*np.pi*self.n*gamma(2*self.n)/(self.bn**(2*self.n)))
        for i in range(len(r_array)):
            out[i] = I0*np.exp(-self.bn*(r_array[i]/Re)**(1/self.n))
        return out        

    def M_enclosed_2D(self,r,Re):
        #* use this function to multiply the total stellar mass
        r = np.atleast_1d(r)
        out = np.zeros_like(r)
        for i in range(len(r)):
            out[i] = gammainc(2*self.n, self.bn*(r[i]/Re)**(1/self.n))
        return out

    #* critical surface density of the gnfw profile
    def Sigma_c(self,z_lens,z_source):
        D_lens = self.cosmo_astropy.angular_diameter_distance(z_lens)
        D_source = self.cosmo_astropy.angular_diameter_distance(z_source)
        D_lens_source = self.cosmo_astropy.angular_diameter_distance_z1z2(z_lens, z_source)
        Sigma_c = c**2/(4*np.pi*G)*D_source/(D_lens*D_lens_source)
        Sigma_c = Sigma_c.to('Msun/kpc^2').value
        return Sigma_c

    #* dimensionless surface density of the gnfw profile
    def kappa(self,r,Re,z_lens,z_source,Mstar):
        Sigma_c = self.Sigma_c(z_lens,z_source)
        return Mstar*self.Sigma(r,Re)/Sigma_c

    #* defelction angle of the gnfw profile
    def alpha(self, r, Re, z_lens, z_source, Mstar):
        #* assume the lens lays in the center of the coordinate
        Sigma_c = self.Sigma_c(z_lens, z_source)
        return Mstar*self.M_enclosed_2D(r,Re)/Sigma_c/r/np.pi




