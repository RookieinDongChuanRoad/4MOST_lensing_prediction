import numpy as np
from scipy.interpolate import splrep, splev,bisplrep,bisplev
from scipy.integrate import quad
from scipy.stats import norm
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from colossus.halo import concentration
from scipy.stats import norm

cosmo_colossus = cosmology.setCosmology('planck13')
import astropy.cosmology
cosmo_astropy = cosmo_colossus.toAstropy()


class OII_emitter:
    '''
    This class used for drawing samples of OII emitter directly from the luminosity function
    '''
    def __init__(self):
        self.logphi0 = -1.89
        self.gamma= -1.96
        self.epsilon = -2.48
        self.zpiv = 1
        self.logL0 = 40.73
        self.beta = 2.61
        self.alpha = -1.25
        # # self.Phi_L_z_interp  = bisplrep(logL_interp,z_interp,self.Phi_L_z(logL_interp,z_interp))
        # self.CDF_L_inv_interp = bisplrep(self.CDF_phi(logL_interp,z_interp),z_interp,logL_interp)
        

    def Phi_L_z(self,logL,z):
        if z < self.zpiv : 
            phi_z = 10**self.logphi0*(1+z)**self.gamma
        else:
            phi_z = 10**self.logphi0*(1+self.zpiv)**(self.gamma+self.epsilon)*(1+z)**(-self.epsilon)
        logL = np.atleast_1d(logL)
        out = np.zeros_like(logL)
        for i in range(len(logL)):
            logLi = logL[i]
            L_Lstar = logLi - self.logL0 - self.beta*np.log10(1+z)
            out[i] = phi_z*(10**L_Lstar)**(1+self.alpha)*np.exp(-10**L_Lstar)
        return out
    
    def CDF_Phi(self,logL,z):
        #* calculate the cumulative distribution function of Phi(L,z)
        logL = np.atleast_1d(logL)
        z = np.atleast_1d(z)
        if len(logL) != len(z):
            raise ValueError('logL and z should have the same length')
        out = np.zeros_like(logL)
        for i in range(len(logL)):
            if logL[i] < 40:
                raise ValueError('logL < 40')
            norm = quad(lambda x: self.Phi_L_z(x,z[i]),40,43)[0]
            out[i] = quad(lambda x: self.Phi_L_z(x,z[i]),40,logL[i])[0]/norm
        return out
    
    def set_interp(self):
        logL_interp = np.linspace(40,43,100)
        z_interp = np.linspace(0.3,2.5,100)
        logL_interp,z_interp = np.meshgrid(logL_interp,z_interp)
        logL_interp = logL_interp.flatten()
        z_interp = z_interp.flatten()
        self.CDF_L_inv_interp = bisplrep(self.CDF_Phi(logL_interp, z_interp),z_interp,logL_interp)
        z_dis_interp = np.linspace(0.3,2.5,100)
        normalization = quad(lambda z: cosmo_astropy.differential_comoving_volume(z).value, 0.3, 2.5)[0]
        cdf_interp = np.zeros_like(z_dis_interp)
        for i in range(len(z_dis_interp)):
            cdf_interp[i] = quad(lambda z: cosmo_astropy.differential_comoving_volume(z).value, 0.3, z_interp[i])[0]/normalization
        cdf_inv_interp = splrep(cdf_interp, z_dis_interp)
        self.z_dis_inv_interp = cdf_inv_interp

    def set_fiducial_mean_N(self):
        z_min = 0.15
        z_max = 2.5
        def dNdz(z):
            phi = quad(lambda x: OII_emitter().Phi_L_z(x,z),40,43)[0]
            return phi*cosmo_astropy.differential_comoving_volume(z).value
        fiducial_mean_N = quad(dNdz,z_min,z_max)[0]
        self.fiducial_mean_N = fiducial_mean_N
        

    def CDF_Phi_inv(self, F,z):
        #* The inverse function of CDF_Phi
        # L = np.logspace(39,43,1000)
        F = np.atleast_1d(F)
        out = np.zeros_like(F)
        for i in range(len(F)):
            out[i] = bisplev(F[i],z,self.CDF_L_inv_interp)
        return out
    
    def CDF_z_dis_inv(self,F):
        F = np.atleast_1d(F)
        out = np.zeros_like(F)
        for i in range(len(F)):
            out[i] = splev(F[i],self.z_dis_inv_interp)
        return out

    def draw_sample(self,num,rmax):
        '''
        Ourput: z,L,x,y (numpy array)
        '''
        #* draw num samples from the luminosity function
        #* 1. draw z from the redshift distribution
        F_z = np.random.rand(num)
        z = self.CDF_z_dis_inv(F_z)
        #* 2. draw L from the luminosity function according to the drawn z
        F_L = np.random.rand(num)
        L = np.zeros(num)
        for i in range(num):
            L[i] = self.CDF_Phi_inv(F_L[i],z[i])
        #* 3. draw the position of the background sources
        r = np.sqrt(np.random.rand(num))*rmax
        theta = np.random.rand(num)*2*np.pi
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        return z,L,x,y
        
    
class ELCOSMOS:
    '''
    This class is used for drawing samples of ELCOSMOS galaxies
    '''
    def __init__(self):
        from astropy.table import Table
        import astropy.units as u
        self.data = Table.read('/Users/liurongfu/Desktop/4MOST_lensing_prediction/codes/data/ELCOSMOS_v1_Jlim.fits')
        self.num_total = len(self.data)
        self.z = self.data['ZPHOT']
        self.oii_line_flux = self.data['OII']
        self.J_VISTA = self.data['J_VISTA']
        self.fiducial_mean_N = self.num_total/(1.38*(u.deg**2)).to('sr').value #* effective arae of ELCOSMOS is 1.38 deg^2

    def draw_sample(self,num,rmax):
        #* draw num samples from the ELCOSMOS catalog
        #* 1. draw num samples from the catalog
        index = np.random.randint(0,self.num_total,num)
        z_sample = self.z[index]
        oii_sample = self.oii_line_flux[index]
        # J_sample = self.J_VISTA[index]
        #* 2. draw the position of the background sources
        r = np.sqrt(np.random.rand(num))*rmax
        theta = np.random.rand(num)*2*np.pi
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        return z_sample,oii_sample,x,y

    
z_min = 0.15
z_max = 2.5
def dNdz(z):
    phi = quad(lambda x: OII_emitter().Phi_L_z(x,z),40,43)[0]
    return phi*cosmo_astropy.differential_comoving_volume(z).value

fiducial_mean_N = quad(dNdz,z_min,z_max)[0]


    















