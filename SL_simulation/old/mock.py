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

class Deflector:
    '''The `deflector` class is used to simulate strong lensing systems. It can generate mock data for lensing systems and calculate various properties such as deflection angles, convergence, and magnification.
    Attributes:
        zmin (float): Minimum redshift for the lensing system.
        zmax (float): Maximum redshift for the lensing system.
        use_cat (bool): Flag to determine whether to use a catalog for generating data.
        cat (str): Name of the catalog to use if `use_cat` is True.
    Methods:
        generate():
            Generates mock data for a strong lensing simulation. If `use_cat` is False, it generates various properties of the lensing system based on random sampling from predefined distributions. If `use_cat` is True, no operations are performed.
        alpha(x, zs):
            Calculates the deflection angle alpha for a given position and source redshift.
                x (array-like): The position at which to calculate the deflection angle.
                zs (float): The redshift of the source.
                array-like: The deflection angle at the given position and source redshift.
        kappa(x, zs):
            Calculates the convergence kappa for a given position and source redshift.
                x (array-like): The position at which to calculate the convergence.
                zs (float): The redshift of the source.
                array-like: The convergence at the given position and source redshift.
        mu_r(x, zs):
            Calculates the radial magnification for a given position and source redshift.
                x (array-like): The position at which to calculate the radial magnification.
                zs (float): The redshift of the source.
                float: The radial magnification at the given position and source redshift.
        mu_t(x, zs):
            Calculates the tangential magnification for a given position and source redshift.
                x (array-like): The position at which to calculate the tangential magnification.
                zs (float): The redshift of the source.
                float: The tangential magnification at the given position and source redshift.
        cal_potential_caustic():
            Calculates the potential caustic properties of the lensing system, including the Einstein radius and caustic area.
    '''
    def __init__(self, zmin = 0, zmax = 0.15, use_cat = False, cat = 'Shark'):
        self.zmin = zmin
        self.zmax = zmax
        self.Red = Redshift(z_min = zmin, z_max = zmax)
        self.use_cat = use_cat
        self.cat = cat

    def generate(self):
        """
        Generates mock data for a strong lensing simulation.
        If `self.use_cat` is False, the method generates various properties of the lensing system
        based on random sampling from predefined distributions. The generated properties include:
        - Number of components (`self.num_components`)
        - Redshift (`self.z`)
        - Ellipticity (`self.q`)
        - Dark matter density slope (`self.gamma_DM`)
        - Stellar mass (`self.logMstar`)
        - Effective radius in kpc (`self.Re_kpc`)
        - Halo mass (`self.logMh`)
        - Concentration parameter (`self.logc`)
        - Stellar population synthesis parameter (`self.alpha_sps`)
        - Effective radius in arcseconds (`self.Re_arcsec`)
        - Virial radius in kpc (`self.rvir_kpc`)
        - Scale radius in kpc (`self.rs_kpc`)
        - Characteristic density (`self.rhos`)
        If `self.use_cat` is True, the method does not perform any operations.
        Note:
        - The method uses various predefined classes and functions such as `Redshift`, `ellipticity`, 
          `gamma_DM`, `SMF_ETG`, `mass_size_relation`, `SHMR`, `concentration`, `cosmo_astropy`, 
          and `deltaVir`.
        - The parameter `self.alpha_sps` is set to a fixed value of 1.1.
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

            self.Re_arcsec = 10**self.logRe_kpc*cosmo_astropy.arcsec_per_kpc_proper(self.z).value
            
            #* some gnfw quantities
            c = 10**self.logc
            rho_c = cosmo_astropy.critical_density(self.z).to('Msun/kpc^3').value#* critical density at z
            self.rvir_kpc = np.cbrt(3*10**self.logMh/(4*np.pi*deltaVir(self.z)*rho_c))#* in kpc
            self.rs_kpc = self.rvir_kpc/c #* in kpc
            self.rhos = (rho_c*deltaVir(self.z)*c**self.gamma_DM*(3-self.gamma_DM))/(3*hyp2f1(3-self.gamma_DM,3-self.gamma_DM,4-self.gamma_DM,-c))
        else:
            pass
    
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
            Sersic_alpha = Sersic.alpha(x,10**self.logRe_kpc, 4 ,self.z,zs,10**self.logMstar)
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
            Sersic_kappa = Sersic.kappa(x,10**self.logRe_kpc, 4 ,self.z,zs,10**self.logMstar)
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

    def find_lens(self, num_density):
        
        circle_area = 1.44*self.radcaustic_sr/self.q#* the area of the circle that is slightly larger than the caustic area
        self.r_circle = np.sqrt((circle_area*u.sr).to('arcsec').value)#* the radius of the circle
        mean_num_bg = circle_area*num_density
        num_bg = poisson.rvs(mean_num_bg)
        self.num_bg = num_bg


class Source:
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

    def __init__(self, use_cat = True, cat = 'ELCOSMOS'):
        self.use_cat = use_cat
        self.cat = cat
        if self.use_cat == True:
            self.cat_data =  Table.read('/Users/liurongfu/Desktop/4MOST_lensing_prediction/codes/data/ELCOSMOS_v1_Jlim.fits')
            self.num_total = len(self.cat_data)
            self.fiducial_mean_N = self.num_total/(1.38*(u.deg**2)).to('sr').value #* effective arae of ELCOSMOS is 1.38 deg^2
            index = np.random.randint(0,self.num_total)
            self.z = self.cat_data['ZPHOT'][index]       
            self.oii_line_flux = self.cat_data['OII'][index]
            self.J_VISTA = self.cat_data['J_VISTA'][index]
            self.frac_r = np.sqrt(np.random.rand())#* the fraction of the radius
            self.theta = np.random.rand()*2*np.pi
        else:
            pass

class Lense:
    def __init__(self, deflector, id):
        """
        Initialize the Lense class with a given Deflector instance.
        Parameters:
            eflector : Deflector
            An instance of the Deflector class.
        """
        self.deflector = deflector
        self.id = id
        num_source = self.deflector.num_bg
        self.sources = [Source() for i in range(num_source)]



    def find_image_pos(self):
        import glafic
        p = [
            cosmo_astropy.Om0, cosmo_astropy.Ode0, cosmo_astropy.w(0), cosmo_astropy.h,
            f'/Users/liurongfu/Desktop/4MOST_lensing_prediction/data/find_image_ELCOSMOS/{self.id}',
            -5, -5, 5, 5, 0.1, 0.1, 5
        ]
        glafic.init(*p)
        glafic.startup_setnum(2, 0, self.deflector.num_bg)
        e = 1 - self.deflector.q

        glafic.set_lens(1, 'gnfw', self.deflector.z, 10**self.deflector.logMh * cosmo_astropy.h, 0, 0, e, 90, self.deflector.rvir_kpc, self.deflector.gamma_DM)
        glafic.set_secondary('nfw_users 0', verb=0)
        glafic.set_lens(2, 'sers', self.deflector.z, 10**self.deflector.logMstar * cosmo_astropy.h, 0, 0, e, 90, 10**self.deflector.logRe_kpc * cosmo_astropy.arcsec_per_kpc_proper(self.deflector.z).value, 4)

        for j in range(self.deflector.num_bg):
            z = self.sources[j].z
            x = self.sources[j].r_circle*np.cos(self.sources[j].theta)
            y = self.sources[j].r_circle*np.sin(self.sources[j].theta)
            glafic.set_point(j + 1, z, x, y)

        glafic.model_init()
        glafic.findimg()

    


        
             

    









            
    