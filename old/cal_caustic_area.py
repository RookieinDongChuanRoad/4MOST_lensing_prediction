from SL_simulation.profiles import gnfw,deV
import numpy as np
import h5py
# import foreground_model as fg
from SL_simulation import foreground_model as fg
import sys
sys.exit()

#* instance of gnfw and deV profiles
gnfw = gnfw()
deV = deV()

#* functions for calculating strong lensing quantities
def alpha(x, rhos,rs,gamma_DM, mstar,re,z_lens, z_source):
    #* here x in kpc
    gnfw_alpha = gnfw.alpha(x,rs,gamma_DM,z_lens,z_source, rhos)
    deV_alpha = deV.alpha(x,re,z_lens, z_source, mstar)
    return gnfw_alpha + deV_alpha

def kappa(x, rhos,rs,gamma_DM, mstar,re,z_lens, z_source):
    #* here x in kpc
    gnfw_kappa = gnfw.kappa(x,rs,gamma_DM,z_lens,z_source, rhos)
    deV_kappa = deV.kappa(x,re,z_lens, z_source, mstar)
    return gnfw_kappa + deV_kappa

def mu_r(x, rhos,rs,gamma_DM, mstar,re,z_lens, z_source):
    #* here x in kpc
    return (1+alpha(x, rhos,rs,gamma_DM, mstar,re,z_lens, z_source)/x - 2*kappa(x, rhos,rs,gamma_DM, mstar,re,z_lens, z_source))**(-1)

def mu_t(x, rhos,rs,gamma_DM, mstar,re,z_lens, z_source):
    #* here x in kpc
    return (1-alpha(x, rhos,rs,gamma_DM, mstar,re,z_lens, z_source)/x)**-1

#* Calculate the caustic area for our foreground sample
file_name = '/root/4MOST_lensing_prediction/codes/data/foreground_sample.h5'

from scipy.optimize import root
import astropy.units as u
def cal_ein_radius(rhos,rs,gamma_DM, mstar,re,z_lens, z_source):
    #* calculate the Einstein radius
    def func(x):
        return alpha(x, rhos,rs,gamma_DM, mstar,re,z_lens, z_source) - x
    sol = root(func, 100).x
    r_ein_arcsec = sol*fg.cosmo_astropy.arcsec_per_kpc_proper(z_lens).value
    return r_ein_arcsec

def cal_caustic_area(rhos,rs,gamma_DM, mstar,re,z_lens, z_source):
    #* calculate the caustic area
    def func(x):
        return 1/mu_r(x, rhos,rs,gamma_DM, mstar,re,z_lens, z_source)
    xrad = root(func, 1e-11).x
    radcaustic_kpc = xrad - alpha(xrad, rhos,rs,gamma_DM, mstar,re,z_lens, z_source)
    radcaustic_arcsec = radcaustic_kpc*fg.cosmo_astropy.arcsec_per_kpc_proper(z_lens).value
    radcaustic_sr = ((radcaustic_arcsec*u.arcsec)**2).to('sr').value
    return radcaustic_sr
    

def cal_caustic_arcsec(rhos,rs,gamma_DM, mstar,re,z_lens, z_source):
    #* calculate the caustic area
    def func(x):
        return 1/mu_r(x, rhos,rs,gamma_DM, mstar,re,z_lens, z_source)
    xrad = root(func, 1e-10).x
    radcaustic_kpc = xrad - alpha(xrad, rhos,rs,gamma_DM, mstar,re,z_lens, z_source)
    radcaustic_arcsec = radcaustic_kpc*fg.cosmo_astropy.arcsec_per_kpc_proper(z_lens).value
    radcaustic_sr = ((radcaustic_arcsec*u.arcsec)**2).to('sr').value
    return radcaustic_arcsec



# if __name__ == '__main__':
#     with h5py.File(file_name,'r+') as f:
#         logMstar = f['logMstar'][()]
#         logRe = f['logRe'][()]
#         z_lens = f['z'][()]
#         rhos = f['rhos'][()]
#         rs = f['rs'][()]
#         gamma_DM = f['gamma_DM'][()]
#         mstar = 10**logMstar
#         re = 10**logRe
#         r_ein_arcsec_ar = np.zeros_like(logMstar)
#         radcaustic_sr_ar = np.zeros_like(logMstar)
#         z_source = 2.5
#         for i in range(len(logMstar)):
#             print('Calculating {}th lens'.format(i))
#             r_ein_arcsec_ar[i] = cal_ein_radius(rhos[i],rs[i],gamma_DM[i], mstar[i],re[i],z_lens[i], z_source)
#             radcaustic_sr_ar[i] = cal_caustic_area(rhos[i],rs[i],gamma_DM[i], mstar[i],re[i],z_lens[i], z_source)

#         print('Saving to file')
#         f.create_dataset('r_ein_arcsec',data=r_ein_arcsec_ar)
#         f.create_dataset('radcaustic_sr',data=radcaustic_sr_ar)




#         f.close()
