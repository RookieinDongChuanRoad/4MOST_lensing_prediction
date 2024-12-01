import numpy as np
from scipy.interpolate import splrep, splev
from scipy.integrate import quad
from scipy.stats import norm
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from colossus.halo import concentration
from scipy.stats import norm

cosmo_colossus = cosmology.setCosmology('planck13')
import astropy.cosmology
cosmo_astropy = cosmo_colossus.toAstropy()

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
        def PDF(logM):
            logM_star = 10.954
            logM_star_err = 0.028
            logphi = -2.994
            logphi_err = 0.025
            alpha = -0.524
            alpha_err = 0.037
            #* Stellar mass function as a function of logM (i.e. probability density function)
            return 10**(logphi) * 10**((logM - logM_star) * (1 + alpha)) * np.exp(-10**(logM - logM_star ))
        self.PDF = PDF
        logM_min = 11
        logM_max = 12.5
        normalization = 1/quad(PDF, logM_min, logM_max)[0]
        x_interpolate = np.linspace(logM_min, logM_max, 1000)
        y_interpolate = [quad(PDF, logM_min, x)[0]*normalization for x in x_interpolate]
        cdf_spline = splrep(x_interpolate, y_interpolate)
        self.CDF_spline = cdf_spline
        # Ensure y_interpolate is sorted and unique
        y_interpolate, unique_indices = np.unique(y_interpolate, return_index=True)
        x_interpolate = x_interpolate[unique_indices]
        cdf_inv_spline = splrep(y_interpolate, x_interpolate)
        self.CDF_inv_spline = cdf_inv_spline
    
    
    
    def CDF(self, logM):
        #* Cumulative distribution function(we would draw random samples from this)
        return splev(logM, self.CDF_spline)
    
    def CDF_inv(self, F):
        #* Inverse of the CDF
        #* Generate a random number F in [1,0], then CDF_inv(F) will give us the corresponding logM
        return splev(F, self.CDF_inv_spline)

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
        self.cosmo = cosmo_astropy
        self.z_min = 0
        self.z_max = 0.15
        self.PDF = lambda z: cosmo_astropy.differential_comoving_volume(z).value
        normalization = 1/quad(self.PDF, self.z_min, self.z_max)[0]
        x_interpolate = np.linspace(self.z_min, self.z_max, 1000)
        y_interpolate = [quad(self.PDF, self.z_min, x)[0]*normalization for x in x_interpolate]
        self.CDF_spline = splrep(x_interpolate, y_interpolate)
        # Ensure y_interpolate is sorted and unique
        y_interpolate, unique_indices = np.unique(y_interpolate, return_index=True)
        x_interpolate = x_interpolate[unique_indices]
        cdf_inv_spline = splrep(y_interpolate, x_interpolate)
        self.CDF_inv_spline = cdf_inv_spline

    # def PDF(self, z):
    #     #* Redshift distribution as a function of z
    #     return self.cosmo.differential_comoving_volume(z).value
    
    def CDF(self, z):
        #* Cumulative distribution function(we would draw random samples from this)
        # normalization = 1/quad(self.PDF, self.z_min, self.z_max)[0]
        # x_interpolate = np.linspace(self.z_min, self.z_max, 1000)
        # y_interpolate = [quad(self.PDF, self.z_min, x)[0]*normalization for x in x_interpolate]
        # return splev(z, splrep(x_interpolate, y_interpolate))
        return splev(z, self.CDF_spline)
    
    def CDF_inv(self, F):
        #* Inverse of the CDF
        #* Generate a random number F in [1,0], then CDF_inv(F) will give us the corresponding z
        # normalization = 1/quad(self.PDF, self.z_min, self.z_max)[0]
        # x_interpolate = np.linspace(self.z_min, self.z_max, 1000)
        # y_interpolate = [quad(self.PDF, self.z_min, x)[0]*normalization for x in x_interpolate]
        # return splev(F, splrep(y_interpolate, x_interpolate))
        return splev(F, self.CDF_inv_spline)
        
#*ellipticity
class q_distribution():
    def __init__(self):
        self.alpha = 6.28
        self.beta = 2.05
        self.PDF = lambda q: q**(6.28)*(1-q)**(2.05)
        normalization = 1/quad(self.PDF, 0, 1)[0]
        x_interpolate = np.linspace(0,1,1000)
        y_interpolate = np.array([quad(self.PDF,0,x)[0]*normalization for x in x_interpolate])
        self.CDF_spline = splrep(x_interpolate, y_interpolate)
        # Ensure y_interpolate is sorted and unique
        y_interpolate, unique_indices = np.unique(y_interpolate, return_index=True)
        x_interpolate = x_interpolate[unique_indices]
        cdf_inv_spline = splrep(y_interpolate, x_interpolate)
        self.CDF_inv_spline = cdf_inv_spline

    # def PDF(self,q):
        #* The probability distribution of axis ratio q 
        # return q**(self.alpha)*(1-q)**(self.beta)

    def CDF(self,q):
        #* The cumulaive distribution function
        # normalizaion = 1/quad(self.PDF, 0, 1)[0]
        # x_interpolate = np.linspace(0,1,1000)
        # y_interpolate = np.array([quad(self.PDF,0,x)[0]*normalizaion for x in x_interpolate])
        # return splev(q,splrep(x_interpolate, y_interpolate))
        return splev(q, self.CDF_spline)
    
    def CDF_inv(self,F):
        #* The inverse of CDF
        #* We can generate a random number F between [0,1], then input into 
        #* this CDF_inv, to get a sample q
        # normalization = 1/quad(self.PDF, 0,1)[0]
        # x_interpolate = np.linspace(0,1,1000)
        # y_interpolate = np.array([quad(self.PDF,0,x)[0]*normalizaion for x in x_interpolate])
        # return splev(F,splrep(x_interpolate, y_interpolate))
        return splev(F, self.CDF_inv_spline)

#* Mass-size relation
class mass_size_relation():
    def __init__(self):
        self.mu_R = 0.99
        self.beta_R = 0.61
        self.sigma_R = 0.20

    def mu_logRe(self,logM):
        return self.mu_R + self.beta_R*(logM - 11.4)
    
    # def draw_logRe(self,logM):
    #     mu = self.mu_logRe(logM)
    #     sigma = self.sigma_R
    #     return np.random.normal(mu,sigma)

    def PDF(self,logRe,logM):
        #* The probability of Re given M
        return norm(loc=self.mu_logRe(logM),scale=self.sigma_R).pdf(logRe)
    
    def CDF(self,logRe,logM):
        #* The cumulative distribution function
        return norm(loc=self.mu_logRe(logM),scale=self.sigma_R).cdf(logRe)
    
    def CDF_inv(self,logM,F):
        #* The inverse of CDF
        #* We can generate a random number F between [0,1], then input into 
        #* this CDF_inv, to get a sample logRe
        return norm(loc=self.mu_logRe(logM),scale=self.sigma_R).ppf(F)
 
#?================================================================================================
#? Halo model

#* Stellar-to-halo mass relation
class SHMR():
    def __init__(self, fix_scatter = False):
        self.fix_scatter = fix_scatter
        self.sigma_logMh = 0.4 #* if we fix the scatter, we can set it to 0.4
        #* parameter from Zu & Mandelbaum 2015
        # self.logM0 = 12.10
        # self.logM1 = 10.31
        # self.beta = 0.33
        # self.delta = 0.42
        # self.gamma = 1.21
        
        #* parameter from Shuntov et al. 2022
        self.logM0 = 12.629
        self.logM1 = 10.855
        self.beta = 0.487
        self.delta = 0.935
        self.gamma = 1.939
        self.sigma_logMstar = 0.268
         #* Here we choose to use a constant scatter for Mh
        def P_Mstar_Mh(logMh,logMstar):
            #* The probability of Mstar given Mh
            logM0 = 12.629
            logM1 = 10.855
            beta = 0.487
            delta = 0.935
            gamma = 1.939
            sigma_logMstar = 0.268
            logmstar_interp_array = np.linspace(10,11.8,1000)
            logmh_interp_array  = logM0 + beta*(logmstar_interp_array - logM1) + (10**(logmstar_interp_array - logM1))**delta / (1 + (10**(logmstar_interp_array - logM1))**(-gamma)) - 0.5
            #* Inverse the model of Shuntov et al. 2022 to get mstar as a function of mh
            spline_mstar_mh = splrep(logmh_interp_array,logmstar_interp_array)
            return norm(loc=splev(logMh,spline_mstar_mh),scale=sigma_logMstar).pdf(logMstar)
        self.P_Mstar_Mh = P_Mstar_Mh
        logMh_array = np.linspace(11.5,18,1000)
        logMstar_ar = np.linspace(10,11.8,1000)
        mean_logMh_ar = np.zeros(len(logMstar_ar))
        std_logMh_ar = np.zeros(len(logMstar_ar))
        for i in range(len(logMstar_ar)):
            distribute = P_Mstar_Mh(logMh_array,logMstar_ar[i]) * mass_function.massFunction(10**logMh_array, 0.1, q_in='M', mdef='vir', q_out='dndlnM', model = 'despali16')
            normalized_distribute = distribute/np.sum(distribute)
            mean_logMh_ar[i] = np.sum(normalized_distribute * logMh_array)
            std_logMh_ar[i] = np.sqrt(np.sum(normalized_distribute * (logMh_array - mean_logMh_ar[i])**2))
        spline_mh_mstar = splrep(logMstar_ar,mean_logMh_ar)
        spline_sigmamh_mstar = splrep(logMstar_ar,std_logMh_ar)
        self.spline_mh_mstar = spline_mh_mstar
        self.spline_sigmamh_mstar = spline_sigmamh_mstar
    
    # def P_Mstar_Mh(self,logMh,logMstar):    
    #     #* The probability of Mstar given Mh
    #     logmstar_interp_array = np.linspace(10,11.8,1000)
    #     logmh_interp_array  = self.logM0 + self.beta*(logmstar_interp_array - self.logM1) + (10**(logmstar_interp_array - self.logM1))**self.delta / (1 + (10**(logmstar_interp_array - self.logM1))**(-self.gamma)) - 0.5
    #     #* Inverse the model of Shuntov et al. 2022 to get mstar as a function of mh
    #     spline_mstar_mh = splrep(logmh_interp_array,logmstar_interp_array)
    #     return norm(loc=splev(logMh,spline_mstar_mh),scale=self.sigma_logMstar).pdf(logMstar)
    
    def mu_Mh_Mstar(self, logMstar):
        #* The mean value of Mh given Mstar
        #* P(Mh|Mstar) = P(Mstar|Mh) * P(Mh)
        # #* P(Mh) is the halo mass function
        return splev(logMstar,self.spline_mh_mstar)    
    
    def sigma_Mh_Mstar(self, logMstar):
        #* The scatter of Mh given Mstar
        return splev(logMstar,self.spline_sigmamh_mstar)
        
    def PDF(self,logMh,logMstar):
        #* The probability of Mh given Mstar
        #* We assume to be a Gaussian distribution around the mean value
        # return norm(self.mu_Mh_Mstar(logMstar),self.sigma_logMh).pdf(logMh)
        if self.fix_scatter:
            return norm(self.mu_Mh_Mstar(logMstar),self.sigma_logMh).pdf(logMh)
        else:
            return norm(self.mu_Mh_Mstar(logMstar),self.sigma_Mh_Mstar(logMstar)).pdf(logMh)
    
    def CDF(self,logMh,logMstar):
        #* The cumulative distribution function
        if self.fix_scatter:
            return norm(self.mu_Mh_Mstar(logMstar),self.sigma_logMh).cdf(logMh)
        else:
            # return norm(self.mu_Mh_Mstar(logMstar),self.sigma_).cdf(logMh)
            return norm(self.mu_Mh_Mstar(logMstar),self.sigma_Mh_Mstar(logMstar)).cdf(logMh)
    
    def CDF_inv(self,logMstar,F):
        #* The inverse of CDF
        #* We can generate a random number F between [0,1], then input into 
        #* this CDF_inv, to get a sample Mh
        if self.fix_scatter:
            return norm(self.mu_Mh_Mstar(logMstar),self.sigma_logMh).ppf(F)
        else:
            # return norm(self.mu_Mh_Mstar(logMstar),self.sigma_logMh).ppf(F)    
            return norm(self.mu_Mh_Mstar(logMstar),self.sigma_Mh_Mstar(logMstar)).ppf(F)

#* inner mass density slope
class inner_slope():
    #* here we only need the scatter, then we can draw the slope from a normal distribution, given the scatter
    def __init__(self):
        self.sigma = 0.2

    def CDF_inv(self,mu_gamma, F):
        #* The inverse of CDF
        #* We can generate a random number F between [0,1], then input into 
        #* this CDF_inv, to get a sample gamma
        return norm(loc=mu_gamma, scale=self.sigma).ppf(F)
    
#* concentration-mass relation
class concentration_mass_relation():
    def __init__(self):
        self.model = 'klypin16_m'
        self.sigma = 0.1

    def mu_logc(self,logM):
        return np.log10(concentration.concentration(10**logM, 'vir', 0.1, model=self.model))
    
    def CDF_inv(self,logM, F):
        #* The inverse of CDF
        #* We can generate a random number F between [0,1], then input into 
        #* this CDF_inv, to get a sample logc
        return norm(loc=self.mu_logc(logM), scale=self.sigma).ppf(F)



































