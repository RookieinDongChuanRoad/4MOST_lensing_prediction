import numpy as np
from SL_simulation.foreground import *
from SL_simulation.profiles import *
from .source import Point_Source
from colossus.cosmology import cosmology
from colossus.halo.mass_so import deltaVir
import astropy.cosmology
from scipy.special import hyp2f1
from scipy.optimize import root
from scipy.stats import poisson
import astropy.units as u
from astropy.table import Table
import glafic

cosmo_colossus = cosmology.setCosmology('planck13')
cosmo_astropy = cosmo_colossus.toAstropy()


class Lense:
    """
    A class representing a gravitational lens system.
    This class handles the initialization, image finding, and image simulation for gravitational lensing systems.
    It uses glafic for lens modeling and image finding, and creates mock images with PSF convolution.
    Parameters
    ----------
    z : float
        Redshift of the lens
    logMh : float
        Log10 of the halo mass
    logc : float
        Log10 of the concentration parameter
    gamma_DM : float
        Dark matter inner slope
    logMstar : float
        Log10 of the stellar mass
    alpha_sps : float
        Stellar population synthesis factor
    Re_arcsec : float
        Effective radius in arcseconds
    q : float
        Axis ratio (minor/major)
    radcaustic_sr : float
        Radial caustic in steradians
    Attributes
    ----------
    prefix : str
        Path to store the images
    z : float
        Lens redshift
    q : float
        Axis ratio
    e : float
        Ellipticity (1-q)
    logMh : float
        Log10 halo mass
    c : float
        Concentration parameter
    gamma_DM : float
        Dark matter inner slope
    c_2 : float
        Modified concentration parameter used in glafic
    logMstar_true : float
        True log stellar mass (includes alpha_sps)
    Re_arcsec : float
        Effective radius in arcseconds
    radcaustic_sr : float
        Radial caustic in steradians
    num_source : int
        Number of background sources
    sources : list
        List of Point_Source objects
    sources_oii : list
        OII line fluxes for each source
    flux_in_fiber : numpy.ndarray
        Array of fluxes within fiber aperture
    r_rein_arcsec : numpy.ndarray
        Array of Einstein radii in arcseconds
    Methods
    -------
    find_image(lens_id)
        Finds lensed images using glafic for all sources
    mock_image(fwhm=0.7, alpha=5)
        Creates mock images with PSF convolution and saves to FITS files
    """

    def __init__(self,z, logMh, logc, gamma_DM, logMstar, alpha_sps, Re_arcsec, q, radcaustic_sr,num_source = -1):

        #* path to store the images
        self.prefix = '/Users/liurongfu/Desktop/4MOST_lensing_prediction/data/ELCOSMOS/'
        #* basic properties
        self.z = z
        self.q = q
        self.e = 1 - q
        #* halo properties
        self.logMh = logMh
        self.c = 10**logc
        self.gamma_DM = gamma_DM
        self.c_2 = self.c/(2 - self.gamma_DM)#* c-2 used in glafic
        #* stellar properties
        self.logMstar_true = logMstar + np.log10(alpha_sps)
        self.Re_arcsec = Re_arcsec
        self.radcaustic_sr = radcaustic_sr
        #* find sources
        if num_source == -1:
            num_density = Point_Source().fiducial_mean_N
            circle_area = 1.44 * self.radcaustic_sr/self.q
            r_circle = np.sqrt((circle_area*u.sr).to('arcsec^2').value)
            mean_number = num_density * circle_area
            self.num_source = poisson.rvs(mean_number)
            self.r_circle = r_circle
        else:
            self.num_source = int(num_source)
            self.r_circle = np.sqrt((1.44 * self.radcaustic_sr/self.q*u.sr).to('arcsec^2').value)
        
        self.sources = [Point_Source() for i in range(self.num_source)] 
        self.sources_oii = [source.oii_line_flux for source in self.sources]
        self.flux_in_fiber = np.zeros(self.num_source)
        self.r_rein_arcsec = np.zeros(self.num_source)
              

    def find_image(self,lens_id):
        """
        Find multiple images of lensed sources using GLAFIC software.
        This method configures and runs GLAFIC to find lensed images for one or more background sources. 
        It uses a combination of gNFW (generalized NFW) and Sersic profiles to model the lens.
        Parameters
        ----------
        lens_id : str or int
            Identifier for the lens system, used in output file naming
        Returns
        -------
        None
            Results are stored in class attributes:
            - r_rein_arcsec : Einstein radius for each source (in arcseconds)
            - Output image files are saved in '{prefix}pos_images/{lens_id}_{source_number}'
        Notes
        -----
        Requires:
        - glafic module for gravitational lensing calculations
        - Pre-configured lens and source parameters in class attributes
        - cosmo_astropy for cosmological calculations
        The lens model combines:
        1. gNFW dark matter halo profile
        2. Sersic profile for stellar component
        Sources are positioned within a circular region defined by r_circle class attribute
        """

        self.lens_id = lens_id
        def run_glafic(source,source_id):
            p = [
            cosmo_astropy.Om0, cosmo_astropy.Ode0, cosmo_astropy.w(0), cosmo_astropy.h,
            self.prefix + 'pos_images/'+ f'{self.lens_id}'+f'_{source_id+1}',
            -5, -5, 5, 5, 0.1, 0.1, 5
            ]
            glafic.init(*p)

            glafic.startup_setnum(2, 0, 1)
            glafic.set_lens(1, 'gnfw', self.z, 10**self.logMh * cosmo_astropy.h, 0, 0, self.e, 90, self.c_2, self.gamma_DM)
            glafic.set_secondary('nfw_users 0', verb=0)
            glafic.set_lens(2, 'sers', self.z, 10**self.logMstar_true * cosmo_astropy.h, 0, 0, self.e, 90, self.Re_arcsec, 4)

            z_source = source.z
            x_source = source.frac_r*np.cos(source.theta)*self.r_circle
            y_source = source.frac_r*np.sin(source.theta)*self.r_circle

            glafic.set_point(1, z_source, x_source, y_source)

            glafic.model_init()
            glafic.findimg()
            self.r_rein_arcsec[source_id] = glafic.calcein_i(z_source)

        if self.num_source > 1:
            for source_id in range(self.num_source):
                source = self.sources[source_id]
                run_glafic(source,source_id)

        else:
            run_glafic(self.sources[0],0)

    def mock_image(self, fwhm = 0.7,alpha = 5):
        """
        Generate mock images of gravitationally lensed sources and convolve them with a Moffat PSF.
        This method creates simulated images of lensed sources by:
        1. Reading image positions and magnifications
        2. Creating pixelized images
        3. Convolving with a Moffat PSF
        4. Calculating flux within fiber aperture
        5. Saving results as FITS files
        Parameters
        ----------
        fwhm : float, optional
            Full width at half maximum of the PSF in arcseconds. Default is 0.7.
        alpha : float, optional
            Power index of the Moffat profile. Default is 5.
        Attributes Modified
        -----------------
        flux_in_fiber : array
            Stores the calculated flux within the fiber aperture for each source.
        Saves
        -----
        FITS files containing:
        - Header with fiber flux and original source flux
        - Raw pixelized image (HDU 1)
        - PSF-convolved image (HDU 2)
        Files are saved as '{lens_id}_{source_id+1}.fits' in the mock_images directory.
        """

        from astropy.convolution import convolve, Moffat2DKernel
        from astropy.io import fits
        
        def convolove_psf(source_id):
            #* read in image position and flux
            pos_image = np.atleast_2d(np.genfromtxt(self.prefix + 'pos_images/'+ f'{self.lens_id}' + f'_{source_id+1}_point.dat'))
            num_image = int(pos_image[0,0])
            x_img = pos_image[1:1+num_image,0]
            y_img = pos_image[1:1+num_image,1]
            magnification = np.abs(pos_image[1:1+num_image,2])
            oii_flux = self.sources_oii[source_id] 
            
            #* set pixel scale
            img_range = 5 # arcsec
            pix_size = 0.1 # arcsec
            gamma_arcsec = fwhm / (2 * np.sqrt(2**(1/alpha) - 1)) # arcsec
            gamma = gamma_arcsec / pix_size
            kernel = Moffat2DKernel(gamma, alpha)

            #* pixelize x,y
            x_img_pix = (img_range/pix_size + np.round(x_img/pix_size)).astype(int)
            y_img_pix = (img_range/pix_size + np.round(y_img/pix_size)).astype(int)

            #* make mock image
            mock_image = np.zeros((int(2*img_range/pix_size)+1, int(2*img_range/pix_size)+1))
            for j in range(num_image):
                mock_image[y_img_pix[j], x_img_pix[j]] = magnification[j]*oii_flux

            #* convolve with Moffat PSF
            convolved_image = convolve(mock_image, kernel)

            #* save convolved image
            x_mesh,y_mesh = np.meshgrid(np.linspace(-img_range, img_range, int(2*img_range/pix_size)+1), np.linspace(-img_range, img_range, int(2*img_range/pix_size)+1))
            r_mesh = np.sqrt(x_mesh**2 + y_mesh**2) 
            fiber_radius = 1.45/2 # the diameter of the fiber is 1.45 arcsec
            in_fiber = r_mesh <= fiber_radius
            flux_in_fiber = np.sum(convolved_image[in_fiber])
            header = fits.Header()
            header['flux_in_fiber'] = flux_in_fiber
            self.flux_in_fiber[source_id] = flux_in_fiber
            header['origin_flux'] = oii_flux

            hdu1 = fits.PrimaryHDU(header=header)
            hdu2 = fits.ImageHDU(data=mock_image)
            hdu3 = fits.ImageHDU(data=convolved_image)
            hdul = fits.HDUList([hdu1, hdu2, hdu3])
            hdul.writeto(self.prefix + 'mock_images/' + f'{self.lens_id}' + f'_{source_id+1}.fits', overwrite=True)        

        if self.num_source > 1:
            for source_id in range(self.num_source):
                convolove_psf(source_id)
        else:
            convolove_psf(0)

    


        

        # glafic.startup_setnum(2, 0, self.deflector.num_bg)
        # e = 1 - self.deflector.q

        # glafic.set_lens(1, 'gnfw', self.deflector.z, 10**self.deflector.logMh * cosmo_astropy.h, 0, 0, e, 90, self.deflector.rvir_kpc, self.deflector.gamma_DM)
        # glafic.set_secondary('nfw_users 0', verb=0)
        # glafic.set_lens(2, 'sers', self.deflector.z, 10**self.deflector.logMstar * cosmo_astropy.h, 0, 0, e, 90, 10**self.deflector.logRe_kpc * cosmo_astropy.arcsec_per_kpc_proper(self.deflector.z).value, 4)

        # for j in range(self.deflector.num_bg):
        #     z = self.sources[j].z
        #     x = self.sources[j].frac_r*np.cos(self.sources[j].theta)*self.deflector.r_circle
        #     y = self.sources[j].frac_r*np.sin(self.sources[j].theta)*self.deflector.r_circle
        #     glafic.set_point(j + 1, z, x, y)

        # glafic.model_init()
        # glafic.findimg()



    # def __init__(self, deflector, id):
    #     """
    #     Initialize the Lense class with a given Deflector instance.
    #     Parameters:
    #         eflector : Deflector
    #         An instance of the Deflector class.
    #     """
    #     self.deflector = deflector
    #     self.id = id
    #     num_source = self.deflector.num_bg
    #     self.sources = [Point_Source() for i in range(num_source)]



    # def find_image_pos(self):
    #     import glafic
    #     p = [
    #         cosmo_astropy.Om0, cosmo_astropy.Ode0, cosmo_astropy.w(0), cosmo_astropy.h,
    #         f'/Users/liurongfu/Desktop/4MOST_lensing_prediction/data/find_image_ELCOSMOS/{self.id}',
    #         -5, -5, 5, 5, 0.1, 0.1, 5
    #     ]
    #     glafic.init(*p)
    #     glafic.startup_setnum(2, 0, self.deflector.num_bg)
    #     e = 1 - self.deflector.q

    #     glafic.set_lens(1, 'gnfw', self.deflector.z, 10**self.deflector.logMh * cosmo_astropy.h, 0, 0, e, 90, self.deflector.rvir_kpc, self.deflector.gamma_DM)
    #     glafic.set_secondary('nfw_users 0', verb=0)
    #     glafic.set_lens(2, 'sers', self.deflector.z, 10**self.deflector.logMstar * cosmo_astropy.h, 0, 0, e, 90, 10**self.deflector.logRe_kpc * cosmo_astropy.arcsec_per_kpc_proper(self.deflector.z).value, 4)

    #     for j in range(self.deflector.num_bg):
    #         z = self.sources[j].z
    #         x = self.sources[j].frac_r*np.cos(self.sources[j].theta)*self.deflector.r_circle
    #         y = self.sources[j].frac_r*np.sin(self.sources[j].theta)*self.deflector.r_circle
    #         glafic.set_point(j + 1, z, x, y)

    #     glafic.model_init()
    #     glafic.findimg()