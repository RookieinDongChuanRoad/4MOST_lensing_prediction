import numpy as np
from scipy.interpolate import splrep,splev
from scipy.integrate import quad
from colossus.cosmology import cosmology
import astropy.cosmology

cosmo_colossus = cosmology.setCosmology('planck13')
cosmo_astropy = cosmo_colossus.toAstropy()

class Redshift():
    def __init__(self,z_min,z_max):
        self.z_min = z_min
        self.z_max = z_max
        self.PDF = lambda z: cosmo_astropy.differential_comoving_volume(z).value
        normalization = 1/quad(self.PDF, self.z_min, self.z_max)[0]
        x_interpolate = np.linspace(self.z_min, self.z_max, 1000)
        y_interpolate = np.array([quad(self.PDF, self.z_min, x)[0]*normalization for x in x_interpolate])
        # make sure y_interpolate is unique
        unique_indices = np.unique(y_interpolate, return_index=True)[1]
        x_interpolate_unique = x_interpolate[unique_indices]
        y_interpolate_unique = y_interpolate[unique_indices]

        self.cdf_spline = splrep(x_interpolate_unique, y_interpolate_unique)
        self.cdf_inv_spline = splrep(y_interpolate_unique, x_interpolate_unique)

    def CDF(self,z):
        return splev(z, self.cdf_spline)
    
    def CDF_inv(self,y):
        return splev(y, self.cdf_inv_spline)
        