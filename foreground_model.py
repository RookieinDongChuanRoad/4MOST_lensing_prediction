import numpy as np
from scipy.interpolate import splrep, splev
from scipy.integrate import quad

#* Stellar mass function 
class SMF_Driver_ETGs():
    #* Stellar mass function for early type galaxies
    #* Parameters from Driver et.al. 2022
    def __init__(self):
        self.logM_star = 10.954
        self.logM_star_err = 0.028
        self.logphi = -2.994
        self.logphi_err = 0.025
        self.alpha = -0.524
        self.alpha_err = 0.037
    
    
    def PDF(self, logM):
        #* Stellar mass function as a function of logM (i.e. probability density function)
        return 10**(self.logphi) * 10**((logM - self.logM_star) * (1 + self.alpha)) * np.exp(-10**(logM - self.logM_star ))
    
   
    def CDF(self, logM):
        #* Cumulative distribution function(we would draw random samples from this)
        logM_min = 11
        logM_max = 12.5
        normalization = 1/quad(self.PDF, logM_min, logM_max)[0]
        x_interpolate = np.linspace(logM_min, logM_max, 1000)
        y_interpolate = [quad(self.PDF, logM_min, x)[0]*normalization for x in x_interpolate]
        return splev(logM, splrep(x_interpolate, y_interpolate))
    
    def CDF_inv(self, F):
        #* Inverse of the CDF
        #* Generate a random number F in [1,0], then CDF_inv(F) will give us the corresponding logM
        logM_min = 11
        logM_max = 12.5
        normalization = 1/quad(self.PDF, logM_min, logM_max)[0]
        x_interpolate = np.linspace(logM_min, logM_max, 1000)
        y_interpolate = [quad(self.PDF, logM_min, x)[0]*normalization for x in x_interpolate]
        return splev(F, splrep(y_interpolate, x_interpolate))

class SMF_Driver_LTGs():
    def __init__(self):
        self.logM_star = 10.436
        self.logM_star_err = 0.038
        self.logphi = -3.332
        self.logphi_err = 0.044
        self.alpha = -1.569
        self.alpha_err = 0.018
   #* To be implemented
   # 
#* Redshift distribution
class Redshift_Distribution():
    def __init__(self):
        from astropy.cosmology import FlatLambdaCDM
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.z_min = 0
        self.z_max = 1.5

    def PDF(self, z):
        #* Redshift distribution as a function of z
        return self.cosmo.differential_comoving_volume(z).value
    
    def CDF(self, z):
        #* Cumulative distribution function(we would draw random samples from this)
        normalization = 1/quad(self.PDF, self.z_min, self.z_max)[0]
        x_interpolate = np.linspace(self.z_min, self.z_max, 1000)
        y_interpolate = [quad(self.PDF, self.z_min, x)[0]*normalization for x in x_interpolate]
        return splev(z, splrep(x_interpolate, y_interpolate))
    
    def CDF_inv(self, F):
        #* Inverse of the CDF
        #* Generate a random number F in [1,0], then CDF_inv(F) will give us the corresponding z
        normalization = 1/quad(self.PDF, self.z_min, self.z_max)[0]
        x_interpolate = np.linspace(self.z_min, self.z_max, 1000)
        y_interpolate = [quad(self.PDF, self.z_min, x)[0]*normalization for x in x_interpolate]
        return splev(F, splrep(y_interpolate, x_interpolate))
        
