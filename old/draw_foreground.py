import numpy as np
import h5py
import foreground_model as fg
import glafic
from colossus.halo.mass_so import deltaVir
from scipy.special import hyp2f1

# cosmo = fg.cosmo_astropy
#* initialize the glafic model
# omega, lamb, weos, hubble = 0.3,0.7, -1.0,0.7
# prefix = '/root/4MOST_lensing_prediction/data/glafic'
# xmin, ymin, xmax, ymax  = -10,-10,10,10
# pix_ext,pix_poi = 0.1,0.1
# maxlev = 5
# glafic.init(omega, lamb, weos, hubble, prefix, xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev)
# #* set glafic secondary parameters
# glafic.set_secondary('flag_hodensity 2',verb = 0)

#* Draw 100000 foreground lenses
sample_size = 100000

#* We need Mstar,Re,z,q, alpha_sps to describe a foreground galaxy
logMstar_array = np.zeros(sample_size)
logRe_array = np.zeros(sample_size)
z_array = np.zeros(sample_size)
q_array = np.zeros(sample_size)
alpha_sps_array = np.zeros(sample_size)

#* We need logMh, c, and gamma_DM to describe a foreground halo
logMh_array = np.zeros(sample_size)
c_array = np.zeros(sample_size)
gamma_DM_array = np.zeros(sample_size)

#* We want the einstein radius and caustic radius for a source at redshift 2.5
# r_caustic_arcsec = np.zeros(sample_size)
# r_einstein_arcsec = np.zeros(sample_size)

#* calculate some gnfw related quantities
rs_array = np.zeros(sample_size)
rhos_array = np.zeros(sample_size)

#* Generate random numbers for each parameter

print('Generating foreground sample...')
#* logMstar
smf_generator = fg.SMF_Driver_ETGs()
F_logMstar = np.random.rand(sample_size)
#* logRe
re_generator = fg.mass_size_relation()
F_logRe = np.random.rand(sample_size)
#* z
z_generator = fg.Redshift_Distribution()
F_z = np.random.rand(sample_size)
#* q
q_generator = fg.q_distribution()
F_q = np.random.rand(sample_size)
#* logMh
logMh_generator = fg.SHMR()
F_logMh = np.random.rand(sample_size)   
#* logc
logc_generator = fg.concentration_mass_relation()
F_logc = np.random.rand(sample_size)
#* gamma_DM
gamma_DM_generator = fg.inner_slope()
F_gamma_DM = np.random.rand(sample_size)

print('done.')
print('Drawing foreground sample.../n')
#* Draw sample
for i in range(sample_size):
    print('Drawing the %d-th sample...'%(i+1))
    #* first draw independent parameters
    z = z_generator.CDF_inv(F_z[i])
    q = q_generator.CDF_inv(F_q[i])
    gamma_DM = gamma_DM_generator.CDF_inv(1.25,F_gamma_DM[i])
    
    z_array[i] = z
    q_array[i] = q
    gamma_DM_array[i] = gamma_DM
    #* then draw dependent parameters
    logMstar = smf_generator.CDF_inv(F_logMstar[i])
    Re = re_generator.CDF_inv(logMstar, F_logRe[i])
    logMh = logMh_generator.CDF_inv(logMstar,F_logMh[i])
    logc = logc_generator.CDF_inv(logMh,F_logc[i])

    logMstar_array[i] = logMstar
    logRe_array[i] = Re
    logMh_array[i] = logMh
    c_array[i] = logc
    #* for alpha_sps, we use a fixed value
    alpha_sps_array[i] = 1.1

    #* Re is in kpc, we want a Re express in arcsec for lensing calculation
    Re_arcsec = 10**Re*fg.cosmo_astropy.arcsec_per_kpc_proper(z).value
    #* c
    c = 10**logc

    #* rs: the scale radius in gnfw profile
    rho_c = fg.cosmo_astropy.critical_density(z).to('Msun/kpc^3').value
    rvir = np.cbrt(3*10**logMh/(4*np.pi*deltaVir(z)*rho_c))
    rs = rvir/c

    #* rhos: the normalization of gnfw profile
    rhos = (rho_c*deltaVir(z)*c**gamma_DM*(3-gamma_DM))/(3*hyp2f1(3-gamma_DM,3-gamma_DM,4-gamma_DM,-c))

    rs_array[i] = rs
    rhos_array[i] = rhos

    # #*set glafic
    # glafic.startup_setnum(2,0,0)
    # glafic.set_lens(1,'gnfw',z,10**logMh*hubble,0,0,1-q,90,c/(2-gamma_DM),gamma_DM)
    # glafic.set_lens(2,'sers',z,10**logMstar*hubble,0,0,1-q,90,Re_arcsec,4)
    # glafic.model_init()

    # #* run glafic
    # Rein_arcsec = glafic.calcein2(2.5,0,0,0)
    # glafic.writecrit(2.5)
    # read_caustic = np.genfromtxt(prefix+'_crit.dat')
    # r_c = np.sqrt(read_caustic[:,2]**2 + read_caustic[:,3]**2)
    # r_caustic_arcsec[i] = np.max(r_c)
    # r_einstein_arcsec[i] = Rein_arcsec

print('Saving foreground sample...')
outname = '/root/4MOST_lensing_prediction/data/foreground_sample.h5'
#* Save the sample
with h5py.File(outname,'w') as f:
    f.create_dataset('logMstar',data=logMstar_array)
    f.create_dataset('logRe',data=logRe_array)
    f.create_dataset('z',data=z_array)
    f.create_dataset('q',data=q_array)
    f.create_dataset('alpha_sps',data=alpha_sps_array)
    f.create_dataset('logMh',data=logMh_array)
    f.create_dataset('c',data=c_array)
    f.create_dataset('gamma_DM',data=gamma_DM_array)
    # f.create_dataset('r_caustic_arcsec',data=r_caustic_arcsec)
    # f.create_dataset('r_einstein_arcsec',data=r_einstein_arcsec)
    f.create_dataset('rs',data=rs_array)
    f.create_dataset('rhos',data=rhos_array)
    f.close
print('done.')


