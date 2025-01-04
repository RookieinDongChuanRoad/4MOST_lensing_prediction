import numpy as np
import h5py
import foreground_model as fg
import background_source as bg
from scipy.stats import poisson
from scipy.interpolate import splrep, splev
from scipy.integrate import quad
import astropy.units as u

foreground_galaxy_name = '/root/4MOST_lensing_prediction/codes/data/foreground_sample_backup.h5'
out_name = '/root/4MOST_lensing_prediction/codes/data/potential_lense_ELCOSMOS.h5'

#* initialize the OII_emitter class
# oii = bg.OII_emitter()
# oii.set_interp()
# oii.set_fiducial_mean_N()
#* draw sample from ELCOSMOS
oii = bg.ELCOSMOS()

#* read the foreground galaxy sample
# foreground_sample = h5py.File(foreground_galaxy_name,'r')
with h5py.File(foreground_galaxy_name,'r') as foreground_sample:
    logMstar = foreground_sample['logMstar'][()]
    logRe = foreground_sample['logRe'][()]
    q = foreground_sample['q'][()]
    r_ein = foreground_sample['r_ein_arcsec'][()]
    z = foreground_sample['z'][()]
    logMh = foreground_sample['logMh'][()]
    rs = foreground_sample['rs'][()]
    gamma_DM = foreground_sample['gamma_DM'][()]
    rhos = foreground_sample['rhos'][()]
    logc = foreground_sample['c'][()]
    alpha_sps = foreground_sample['alpha_sps'][()]
    # logMstar_true = foreground_sample['logMstar_true'][()]
    #* the maximum volume a background source to be lensed can locate in
    radcaustic_sr = foreground_sample['radcaustic_sr'][()]
    radcaustic_sr[radcaustic_sr > 1e-2] = 1e-12

#* draw a circle that is slightly larger than the caustic area, and calculate the mean number of background sources in this circle
a_circle = 1.44*radcaustic_sr/q
mean_num_bg = a_circle*oii.fiducial_mean_N

#* draw the number of background sources in the caustic area from a Poisson distribution
num_bg = poisson.rvs(mean_num_bg)

#* get the index that the background sources is non-zero
ind = num_bg != 0
print('number of background sources:',np.sum(num_bg[ind]))
num_bg_havebg = num_bg[ind]
logMstar_havebg = logMstar[ind]
# logMstar_true_havebg = logMstar_true[ind]
logRe_havebg = logRe[ind]
q_havebg = q[ind]
r_ein_havebg = r_ein[ind]
z_foreground = z[ind]
logMh_havebg = logMh[ind]
rs_havebg = rs[ind]
gamma_DM_havebg = gamma_DM[ind]
rhos_havebg = rhos[ind]
logc_havebg = logc[ind]
alpha_sps_havebg = alpha_sps[ind]
radcaustic_sr_havebg = radcaustic_sr[ind]
a_circle = a_circle[ind]
sample_id = np.arange(len(num_bg_havebg)).astype(str)

#* determine the redshift and the luminosity of the background sources, as well its position

# #* 1. find the redshift distribution of the background sources, and the inverse of the CDF
# z_interp = np.linspace(0.15, 2.5, 100)
# normalization = quad(lambda z: fg.cosmo_astropy.differential_comoving_volume(z).value, 0.15, 2.5)[0]
# cdf_interp = np.zeros_like(z_interp)
# for i in range(len(z_interp)):
#     cdf_interp[i] = quad(lambda z: fg.cosmo_astropy.differential_comoving_volume(z).value, 0.15, z_interp[i])[0]/normalization
# cdf_inv_interp = splrep(cdf_interp, z_interp)

#* draw the redshift of the background sources, then draw L from the luminosity function according to the drawn z
with h5py.File(out_name,'w') as f:
    gamma_generator = fg.inner_slope()
    for i in range(len(num_bg_havebg)):
        id = sample_id[i]
        group = f.create_group(id)
        group.attrs['num_bg'] = num_bg_havebg[i]
        group.attrs['logMstar'] = logMstar_havebg[i]
        group.attrs['logRe'] = logRe_havebg[i]
        group.attrs['q'] = q_havebg[i]
        group.attrs['r_ein'] = r_ein_havebg[i]
        group.attrs['z_foreground'] = z_foreground[i]
        group.attrs['logMh'] = logMh_havebg[i]
        group.attrs['rs'] = rs_havebg[i]
        #* correct for some uncorrect gamma_DM
        gamma_DM = gamma_DM_havebg[i]
        while gamma_DM < 2:
            gamma_DM = gamma_generator.CDF_inv(1.25,np.random.rand())
        group.attrs['gamma_DM'] = gamma_DM
        group.attrs['rhos'] = rhos_havebg[i]
        group.attrs['logc'] = logc_havebg[i]
        group.attrs['alpha_sps'] = alpha_sps_havebg[i]
        # group.attrs['radcaustic_sr'] = radcaustic_sr_havebg[i]
        cs_sr = radcaustic_sr_havebg[i]*u.sr
        cs_arcsec = np.sqrt(cs_sr.to('arcsec^2').value)
        group.attrs['radcaustic_arcsec'] = cs_arcsec

        #* draw sample in r_circle
        a_circ_sr = a_circle[i]*u.sr
        r_circ_arcsec = np.sqrt(a_circ_sr.to('arcsec^2').value)
        group.attrs['r_circle'] = r_circ_arcsec

        z_background, L_background, x_background, y_background = oii.draw_sample(num_bg_havebg[i], r_circ_arcsec)
        # z_background = np.zeros(num_bg_havebg[i])
        # L_background = np.zeros(num_bg_havebg[i])
        # x_background = np.zeros(num_bg_havebg[i])
        # y_background = np.zeros(num_bg_havebg[i])

        # for j in range(num_bg_havebg[i]):
        #     #* draw the redshift of the background sources
        #     ran_z = np.random.rand()
        #     z_i = splev(ran_z, cdf_inv_interp)

        #     #* draw L from the luminosity function according to the drawn z
        #     ran_L = np.random.rand()
        #     L_i = oii.CDF_Phi_inv(ran_L, z_i)

        #     #* draw the position of the background sources
        #     r_max = r_circ_arcsec
        #     r = np.sqrt(np.random.rand())*r_max
        #     theta = np.random.rand()*2*np.pi
        #     x = r*np.cos(theta)
        #     y = r*np.sin(theta)

        #     #* save the drawn z, L, x, y
        #     z_background[j] = z_i
        #     L_background[j] = L_i
        #     x_background[j] = x
        #     y_background[j] = y

        group.create_dataset('z_background', data=z_background)
        group.create_dataset('L_background', data=L_background)
        group.create_dataset('x_background', data=x_background)
        group.create_dataset('y_background', data=y_background)










