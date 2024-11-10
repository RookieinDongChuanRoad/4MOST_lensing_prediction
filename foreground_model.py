import numpy as np
from scipy.interpolate import splrep, splev
from scipy.integrate import quad

#?================================================================================================
#? Galaxy model
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
        
#*ellipticity
class q_distribution():
    def __init__(self):
        self.alpha = 6.28
        self.beta = 2.05

    def PDF(self,q):
        #* The probability distribution of axis ratio q 
        return q**(self.alpha)*(1-q)**(self.beta)

    def CDF(self,q):
        #* The cumulaive distribution function
        normalizaion = 1/quad(self.PDF, 0, 1)[0]
        x_interpolate = np.linspace(0,1,1000)
        y_interpolate = np.array([quad(self.PDF,0,x)[0]*normalizaion for x in x_interpolate])
        return splev(q,splrep(x_interpolate, y_interpolate))
    
    def CDF_inv(self,F):
        #* The inverse of CDF
        #* We can generate a random number F between [0,1], then input into 
        #* this CDF_inv, to get a sample q
        normalization = 1/quad(self.PDF, 0,1)[0]
        x_interpolate = np.linspace(0,1,1000)
        y_interpolate = np.array([quad(self.PDF,0,x)[0]*normalizaion for x in x_interpolate])
        return splev(F,splrep(x_interpolate, y_interpolate))

#* Mass-size relation
class mass_size_relation():
    def __init__(self):
        self.mu_R = 0.99
        self.beta_R = 0.61
        self.sigma_R = 0.20

    def mu_logRe(self,logM):
        return self.mu_R + self.beta_R*(logM - 11.4)
    
    def draw_logRe(self,logM):
        mu = self.mu_logRe(logM)
        sigma = self.sigma_R
        return np.random.normal(mu,sigma)
 
#?================================================================================================
#? Halo model

#* Stellar-to-halo mass relation
class SHMR():
    def __init__(self):
        #* parameter from Zu & Mandelbaum 2015
        self.logM0 = 12.10
        self.logM1 = 10.31
        self.beta = 0.33
        self.delta = 0.42
        self.gamma = 1.21
        

    def mu_logMh(self,logM):
        return self.logM0 + self.beta*(logM - self.logM1) + (10**(logM - self.logM1))**self.delta / (1 + (10**(logM - self.logM1))**(-self.gamma)) - 0.5

#* inner mass density slope
class inner_slope():
    #* here we only need the scatter, then we can draw the slope from a normal distribution, given the scatter
    def __init__(self):
        self.sigma = 0.2

    def CDF_inv(self,mu_gamma, F):
        #* The inverse of CDF
        #* We can generate a random number F between [0,1], then input into 
        #* this CDF_inv, to get a sample gamma
        from scipy.stats import norm
        return norm.ppf(F,loc=mu_gamma, scale=self.sigma)
    
#* concentration-mass relation
class concentration_mass_relation():
    def __init__(self):
        self.model = 'Klypin16_m'
        self.sigma = 0.1

    def mu_logc(self,logM):
        from colossus.halo import concentration
        from colossus.cosmology import cosmology
        cosmology.setCosmology('planck13')
        return np.log10(concentration.concentration(10**logM, 'vir', 0.1, model=self.model))
    
    def CDF_inv(self,logM, F):
        #* The inverse of CDF
        #* We can generate a random number F between [0,1], then input into 
        #* this CDF_inv, to get a sample c
        from scipy.stats import norm
        return norm.ppf(F,loc=self.mu_logc(logM), scale=self.sigma)



































