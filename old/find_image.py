import numpy as np
import h5py
import glafic 
from astropy.cosmology import FlatLambdaCDM
import foreground_model as fg

with h5py.File('/root/4MOST_lensing_prediction/codes/data/potential_lense.h5', 'r') as galaxies:
    # for i in range(len(galaxies)):
    for i in range(882,len(galaxies)): #* start from the 754th
        id = str(i)
        sample = galaxies[id]
        alpha_sps = sample.attrs['alpha_sps']
        gamma_DM = sample.attrs['gamma_DM']
        logMh = sample.attrs['logMh']
        logMstar = sample.attrs['logMstar']
        logRe = sample.attrs['logRe']
        c = 10**sample.attrs['logc']
        num_bg = sample.attrs['num_bg']
        q = sample.attrs['q']
        r_ein = sample.attrs['r_ein']
        radcaustic_arcsec = sample.attrs['radcaustic_arcsec']
        rhos = sample.attrs['rhos']
        rs = sample.attrs['rs']
        z_foreground = sample.attrs['z_foreground']
        logMstar_true = logMstar+np.log10(alpha_sps)
        rs = rs*fg.cosmo_astropy.arcsec_per_kpc_proper(z_foreground).value

        #* run glafic to find the images
        p = [fg.cosmo_astropy.Om0, fg.cosmo_astropy.Ode0, fg.cosmo_astropy.w(0), fg.cosmo_astropy.h, '/root/4MOST_lensing_prediction/data/find_image/'+ id, -5, -5, 5, 5, 0.1, 0.1, 5]
        glafic.init(*p)

        #* three number in startup_setnum are the number of lenses, extended sources, and point sources
        #* here we only consider point sources
        glafic.startup_setnum(2,0,num_bg)
        e = 1-q

        # if gamma_DM >= 1.9:
        #     glafic.set_lens(1,'gnfw', z_foreground,10**logMh*fg.cosmo_astropy.h,0,0,e,90,rs, gamma_DM)
        #     glafic.set_secondary('nfw_users 1',verb = 0)
        # else:
        glafic.set_lens(1,'gnfw', z_foreground,10**logMh*fg.cosmo_astropy.h,0,0,e,90,c/(2-gamma_DM), gamma_DM)
        glafic.set_secondary('nfw_users 0',verb = 0)     
        glafic.set_lens(2,'sers',z_foreground,10**logMstar_true*fg.cosmo_astropy.h,0,0,e,90,10**logRe*fg.cosmo_astropy.arcsec_per_kpc_proper(z_foreground).value, 4)

        glafic.set_secondary('nfw_users 1',verb = 0)
        
        for j in range(num_bg):
            z = sample['z_background'][j]
            x = sample['x_background'][j]
            y = sample['y_background'][j]
            glafic.set_point(j+1, z, x, y)
        
        glafic.model_init()

        glafic.findimg()