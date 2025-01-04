import numpy as np
from SL_simulation.foreground import *
from SL_simulation.profiles import *
from colossus.cosmology import cosmology
from colossus.halo.mass_so import deltaVir
import astropy.cosmology
from scipy.special import hyp2f1
from scipy.optimize import root
from scipy.stats import poisson
import astropy.units as u
from astropy.table import Table

cosmo_colossus = cosmology.setCosmology('planck13')
cosmo_astropy = cosmo_colossus.toAstropy()
cat_data =  Table.read('/Users/liurongfu/Desktop/4MOST_lensing_prediction/codes/data/ELCOSMOS_v1_Jlim.fits')

class Point_Source:
    """
    A class to represent a source for lensing prediction simulations.
    Attributes
    ----------
    use_cat : bool
        A flag indicating whether to use a catalog (default is True).
    cat : str
        The name of the catalog to use (default is 'ELCOSMOS').
    cat_data : astropy.table.Table
        The data from the catalog (only if use_cat is True).
    num_total : int
        The total number of entries in the catalog (only if use_cat is True).
    fiducial_mean_N : float
        The fiducial mean number of sources per steradian (only if use_cat is True).
    Methods
    -------
    generate():
        Generates a random source from the catalog if use_cat is True.
    """

    def __init__(self):
        self.cat_data = cat_data
        self.num_total = len(self.cat_data)
        self.fiducial_mean_N = self.num_total/(1.38*(u.deg**2)).to('sr').value #* effective arae of ELCOSMOS is 1.38 deg^2
        index = np.random.randint(0,self.num_total)
        self.z = self.cat_data['ZPHOT'][index]       
        self.oii_line_flux = self.cat_data['OII'][index]
        self.J_VISTA = self.cat_data['J_VISTA'][index]
        self.frac_r = np.sqrt(np.random.rand())#* the fraction of the radius
        self.theta = np.random.rand()*2*np.pi
        