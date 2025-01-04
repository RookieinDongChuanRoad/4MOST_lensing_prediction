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
from .source import Point_Source

cosmo_colossus = cosmology.setCosmology('planck13')
cosmo_astropy = cosmo_colossus.toAstropy()

#* set the default redshift range
zmin = 0
zmax = 0.3
Red  = Redshift(z_min = zmin, z_max = zmax)

#* get number density 
num_density = Point_Source().fiducial_mean_N


class Deflector:
    """A class representing a gravitational lens deflector with multiple components.
    This class models a gravitational lens system that can include both a generalized NFW (GNFW) 
    dark matter halo and a Sersic profile for the stellar component. It can either use predefined 
    catalog values or generate parameters from various distribution functions.
        Red : object
            Redshift distribution object for the deflector.
        use_cat : bool
            Flag indicating whether to use catalog values (True) or generate parameters (False).
        cat : str
            Name of the catalog to use if use_cat is True (default is 'Shark').
        num_components : int
            Number of components in the lens model (typically 2 for GNFW + Sersic).
        z : float
            Redshift of the deflector.
        q : float
            Ellipticity parameter.
        gamma_DM : float
            Dark matter density slope.
        logMstar : float
            Log10 of the stellar mass in solar masses.
        logRe_kpc : float
            Log10 of the effective radius in kpc.
        logMh : float
            Log10 of the halo mass in solar masses.
        logc : float
            Log10 of the concentration parameter.
        alpha_sps : float
            Stellar population synthesis parameter.
        Re_arcsec : float
            Effective radius in arcseconds.
        rvir_kpc : float
            Virial radius in kpc.
        rs_kpc : float
            Scale radius in kpc.
        rhos : float
            Characteristic density.
            Einstein radius in arcseconds.
            Caustic radius in kpc.
            Caustic radius in arcseconds.
            Caustic area in steradians.
    Methods:
        alpha(x, zs, test_ser=0, test_gnfw=0):
            Calculate the deflection angle at position x for source redshift zs.
        kappa(x, zs):
            Calculate the convergence at position x for source redshift zs.
        mu_r(x, zs):
            Calculate the radial magnification factor.
        mu_t(x, zs):
            Calculate the tangential magnification factor.
        cal_potential_caustic():
            Calculate the potential caustic properties.
        set_self():
            Initialize the deflector properties either from distributions or catalog."""
    def __init__(self, red = Red,use_cat = False, cat = 'Shark'):
        self.Red = Red
        self.use_cat = use_cat
        self.cat = cat

    def alpha(self, x,zs,test_ser = 0, test_gnfw = 0):
        """
        Calculate the deflection angle alpha for a given position and source redshift.
        Parameters:
        x : array-like
            The position at which to calculate the deflection angle.
        zs : float
            The redshift of the source.
        Returns:
        array-like
            The deflection angle at the given position and source redshift.
        Notes:
        - If the number of components is 2, the deflection angle is calculated as the sum of the GNFW and Sersic components.
        - If the number of components is not 2, the function currently does nothing.
        """
        
        if self.num_components == 2:
            gnfw_alpha = gnfw.alpha(x,self.rs_kpc,self.gamma_DM,self.z,zs,self.rhos)
            Sersic_alpha = Sersic.alpha(x,10**self.logRe_kpc, 4 ,self.z,zs,10**self.logMstar_true)
            if test_ser == 1:
                return Sersic_alpha
            elif test_gnfw == 1:
                return gnfw_alpha
            else:
                return gnfw_alpha + Sersic_alpha
        else:
            pass

    def kappa(self, x,zs):
        """
            Calculate the convergence (kappa) at a given position and source redshift.
            Parameters:
            -----------
            x : float or array-like
                The position(s) at which to calculate the convergence.
            zs : float
                The redshift of the source.
            Returns:
            --------
            float or array-like
                The calculated convergence at the given position(s) and source redshift.
            Notes:
            ------
            - If the number of components is 2, the convergence is calculated as the sum of the GNFW and Sersic components.
        - If the number of components is not 2, the function currently does nothing.
        """
        
        if self.num_components == 2:
            gnfw_kappa = gnfw.kappa(x,self.rs_kpc,self.gamma_DM,self.z,zs,self.rhos)
            Sersic_kappa = Sersic.kappa(x,10**self.logRe_kpc, 4 ,self.z,zs,10**self.logMstar_true)
            return gnfw_kappa + Sersic_kappa
        else:
            pass
    
    def mu_r(self, x,zs):
        """
        Calculate the magnification factor (mu_r) for a given position and source redshift.
        Parameters:
        x (float): The position at which to calculate the magnification factor.
        zs (float): The redshift of the source.
        Returns:
        float: The magnification factor at the given position and source redshift.
        """

        return (1+self.alpha(x,zs)/x - 2*self.kappa(x,zs))**(-1)

    def mu_t(self, x,zs):
        """
        Calculate the magnification factor (mu_t) for a given position and source redshift.
        Parameters:
        x (float): The position variable.
        zs (float): The source redshift.
        Returns:
        float: The magnification factor.
        """
        
        return (1-self.alpha(x,zs)/x)**-1

    def cal_potential_caustic(self):
        """
        Calculate the potential caustic for a given lensing system.
        This method calculates the Einstein radius and the caustic area for a lensing system
        with a source located at a redshift of zs = 2.5. The results are stored as attributes
        of the instance.
        Attributes:
        -----------
        r_ein_arcsec : float
            The Einstein radius in arcseconds.
        radcaustic_kpc : float
            The radius of the caustic in kiloparsecs.
        radcaustic_arcsec : float
            The radius of the caustic in arcseconds.
        radcaustic_sr : float
            The area of the caustic in steradians.
        """

        zs = 2.5#* Assume an object located at zs = 2.5

        def func_ein_radius(x):
            return self.alpha(x,zs) - x
        sol_ein_radius = root(func_ein_radius, 100).x
        self.r_ein_arcsec = sol_ein_radius*cosmo_astropy.arcsec_per_kpc_proper(self.z).value

        def func_caustic_area(x):
            return 1/self.mu_r(x,zs)
        self.xrad = root(func_caustic_area, 1e-10).x
        self.radcaustic_kpc = self.xrad - self.alpha(self.xrad,zs)
        self.radcaustic_arcsec = np.abs(self.radcaustic_kpc*cosmo_astropy.arcsec_per_kpc_proper(self.z).value)
        self.radcaustic_sr = ((self.radcaustic_arcsec*u.arcsec)**2).to('sr').value
        if self.radcaustic_sr > 1e-2:
            self.radcaustic_sr = 1e-12

    def set_self(self):
        """
        Sets the properties of the deflector object based on random sampling from various distributions.
        If `use_cat` is False, the following properties are set:
        - `num_components`: Number of components, set to 2.
        - `z`: Redshift, sampled from the cumulative distribution function (CDF) of `Red`.
        - `q`: Ellipticity, sampled from the CDF of `ellipticity`.
        - `gamma_DM`: Dark matter density slope, sampled from the CDF of `gamma_DM` until it is less than or equal to 2.
        - `logMstar`: Logarithm of the stellar mass, sampled from the CDF of `SMF_ETG`.
        - `logRe_kpc`: Logarithm of the effective radius in kpc, sampled from the CDF of `mass_size_relation` based on `logMstar`.
        - `logMh`: Logarithm of the halo mass, sampled from the CDF of `SHMR` based on `logMstar`.
        - `logc`: Logarithm of the concentration, sampled from the CDF of `concentration` based on `logMh`.
        - `alpha_sps`: Stellar population synthesis parameter, set to a fixed value of 1.1.
        - `Re_arcsec`: Effective radius in arcseconds, calculated from `logRe_kpc` and the cosmology.
        - `rvir_kpc`: Virial radius in kpc, calculated from `logMh`, `z`, and the critical density.
        - `rs_kpc`: Scale radius in kpc, calculated from `rvir_kpc` and `logc`.
        - `rhos`: Characteristic density, calculated from `rho_c`, `deltaVir`, `c`, and `gamma_DM`.
        Additionally, the potential caustic is calculated by calling `cal_potential_caustic()`.
        If `use_cat` is True, no properties are set.
        """
        
        if self.use_cat == False :
            self.num_components = 2
            
            self.z = self.Red.CDF_inv(np.random.rand())
            
            self.q = ellipticity.CDF_inv(np.random.rand())
            self.gamma_DM = gamma_DM.CDF_inv(np.random.rand())
            while self.gamma_DM > 2:
                self.gamma_DM = gamma_DM.CDF_inv(np.random.rand())
            
            self.logMstar = SMF_ETG.CDF_inv(np.random.rand())
            self.logRe_kpc = mass_size_relation.CDF_inv(np.random.rand(), self.logMstar)
            self.logMh = SHMR.CDF_inv(np.random.rand(), self.logMstar)
            self.logc = concentration.CDF_inv(self.logMh, np.random.rand())
            
            self.alpha_sps = 1.1#* use a fixed value
            self.logMstar_true = self.logMstar+ np.log10(self.alpha_sps)

            self.Re_arcsec = 10**self.logRe_kpc*cosmo_astropy.arcsec_per_kpc_proper(self.z).value
            
            #* some gnfw quantities
            c = 10**self.logc
            rho_c = cosmo_astropy.critical_density(self.z).to('Msun/kpc^3').value#* critical density at z
            self.rvir_kpc = np.cbrt(3*10**self.logMh/(4*np.pi*deltaVir(self.z)*rho_c))#* in kpc
            self.rs_kpc = self.rvir_kpc/c #* in kpc
            self.rhos = (rho_c*deltaVir(self.z)*c**self.gamma_DM*(3-self.gamma_DM))/(3*hyp2f1(3-self.gamma_DM,3-self.gamma_DM,4-self.gamma_DM,-c))

            #* calculate the potential caustic
            self.cal_potential_caustic()
            #* calculate num of soruces
            circle_area = 1.44*self.radcaustic_sr/self.q
            mean_number = circle_area*num_density
            self.num_source = poisson.rvs(mean_number)

        else:
            pass
    

    # def find_lens(self, num_density):
        
    #     circle_area = 1.44*self.radcaustic_sr/self.q#* the area of the circle that is slightly larger than the caustic area
    #     self.r_circle = np.sqrt((circle_area*u.sr).to('arcsec^2').value)#* the radius of the circle
    #     mean_num_bg = circle_area*num_density
    #     num_bg = poisson.rvs(mean_num_bg)
    #     self.num_bg = num_bg


