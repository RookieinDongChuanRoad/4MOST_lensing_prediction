import os
import sys
sys.path.append('/root/software/pygalaxev/')
import pygalaxev
import pygalaxev_cosmology
from pygalaxev_cosmology import c as csol, L_Sun, Mpc
import numpy as np
import h5py
from scipy.interpolate import splrep, splev,splint

# !creates CSP models on a grid of stellar population parameters using galaxev

# *selects the stellar template library:
# *Low-resolution 'BaSeL' library, Chabrier IMF
ssp_dir = '/root/software/BC03/BaSeL3.1_Atlas/Chabrier_IMF/'
tempname = 'lr_BaSeL'
# *work directory
work_dir = '/root/4MOST_lensing_prediction/data/SPS'

# grid_file = h5py.File(work_dir+'/BaSeL_Chabrier_sed_grid.hdf5', 'r')
#* filter directory
filtdir = '/root/software/pygalaxev/filters/'
bands = ['J','Ks']

# *Using Padova 1994 tracks.
Z_given_code = {'m22':0.0001, 'm32':0.0004, 'm42':0.004, 'm52':0.008, 'm62': 0.02, 'm72': 0.05, 'm82': 0.1}

mu = 0.3 # *fraction of attenuation due to diffuse interstellar medium (fixed)
epsilon = 0. # *gas recycling (no recycling if set to zero) (fixed)

nwav = 2023 # *size of wavelength grid (can be looked up by running 'csp_galaxev' on any .ised file of the spectral library)

nZ = len(Z_given_code) # *size of metallicity grid
ntau_V = 21 # *size of grid in dust attenuation

tau_low = 1e-2
tau_high = 2.
tau_V_grid = np.logspace(-2., np.log10(2.), ntau_V
                         )
Z_grid = []

grid = np.zeros((nZ, ntau_V, nwav))

tmpname = work_dir+'/tmp.in'

for m in range(2,9):#* loop over metallicities
    Zcode = 'm%d2' % m
    Z = Z_given_code[Zcode]
    Z_grid.append(Z)

    for tV in range(ntau_V):
        isedname = ssp_dir+'/bc2003_%s_%s_chab_ssp.ised'%(tempname, Zcode)
        cspname = 'bc03_Z=%6.4f_tau=%5.3f_tV=%5.3f_mu=%3.1f_eps=%5.3f'%(Z, 1., tau_V_grid[tV], mu, epsilon)

        pygalaxev.run_csp_galaxev(isedname, cspname,sfh='SSP', sfh_pars = 1. , tau_V=tau_V_grid[tV], mu=0.3, epsilon=0., work_dir=work_dir)

        # Create the mass normalization models
        massname = work_dir+'/'+cspname+'.mass'
        d = np.loadtxt(massname)
        mass_spline = splrep(d[:, 0], d[:, 10], k=3, s=0) #using the sum of M*_liv+M_rem to renormalize the mass

        # extracts SEDs on age grid
        oname = work_dir+'/'+cspname+'_agegrid.sed'
        pygalaxev.create_galaxevpl_config(tmpname, work_dir+'/'+cspname+'.ised', oname, 10)
        os.system('$bc03/galaxevpl < %s'%tmpname)

        f = open(oname, 'r')
        wsed = np.loadtxt(f)
        f.close()

        wave = wsed[:, 0]
        flux = wsed[:, 1]

        #* renormalize the mass
        logAge = np.log10(10)+9.
        mass = splev(logAge, mass_spline)
        sed = flux/mass

        grid[m-2, tV, :] = sed

        #* Clean up
        os.system('rm %s'%oname)

Z_grid = np.array(Z_grid)

#* Save the grid
grid_file = h5py.File(work_dir+'/BaSeL_Chabrier_sed_grid.hdf5', 'w')
grid_dset = grid_file.create_dataset('sed_grid', data=grid)
grid_dset.attrs['units'] = 'Llambda (in units of L_Sun/Angstrom) for 1M_Sun (living + remnants)'
grid_dset.attrs['axis_0'] = 'Metallicity'
grid_dset.attrs['axis_1'] = 'tau_V'
grid_dset.attrs['axis_2'] = 'Wavelength'

grid_file.create_dataset('Z_grid', data=Z_grid)
grid_file.create_dataset('tau_V_grid', data=tau_V_grid)
grid_file.create_dataset('wave', data=wave)

#! calculate the magnitude from a grid a M*
grid_file = h5py.File(work_dir+'/BaSeL_Chabrier_sed_grid.hdf5', 'r')
filtdir = '/root/software/pygalaxev/filters/'
bands = ['J','Ks']

nMstar = 31
Mstar_grid = np.linspace(9, 12, nMstar)

redshift = 0.15
Dlum = pygalaxev_cosmology.Dlum(redshift)#* luminosity distance in Mpc
output_file = h5py.File(work_dir+'/vst_mags_grid_z=%6.4f_fast.hdf5'%redshift, 'w')


wave = grid_file['wave'][()]

wave_obs = wave*(1+redshift)
sed_grid = grid_file['sed_grid'][()]
nZ, ntau_V, nwav = sed_grid.shape

for band in bands:
    filtname = filtdir+'VISTA_%s.res'%band

    f = open(filtname, 'r')
    filt_wave, filt_t = np.loadtxt(f, unpack=True)
    f.close()

    filt_spline = splrep(filt_wave, filt_t)

    wmin_filt, wmax_filt = filt_wave[0], filt_wave[-1]
    cond_filt = (wave_obs>=wmin_filt)&(wave_obs<=wmax_filt)
    nu_cond = np.flipud(cond_filt)

    #* evaluate the filter response at the wavelengths of the spectrum
    response = splev(wave_obs[cond_filt], filt_spline)
    nu_filter = csol*1e8/wave_obs[cond_filt]

    # flips arrays
    response = np.flipud(response)
    nu_filter = np.flipud(nu_filter)

    # filter normalization
    bp = splrep(nu_filter, response/nu_filter, s=0, k=1)
    bandpass = splint(nu_filter[0], nu_filter[-1], bp)

    mag_grid = np.zeros((nMstar,nZ, ntau_V))
    for i in range(len(Mstar_grid)):
        logmstar = Mstar_grid[i]
        for j in range(nZ):
            print('%s band, Z=%6.4f'%(band, grid_file['Z_grid'][j]))
            for k in range(ntau_V):

                    llambda = sed_grid[j, k, :]

                    flambda_obs = llambda*L_Sun/(4.*np.pi*(Dlum*Mpc)**2)/(1.+redshift) # observed specific flux in erg/s/cm^2/AA
                    fnu = flambda_obs * wave_obs**2 / csol * 1e-8 # F_nu in cgs units

                    nu_obs = np.flipud(csol/wave_obs*1e8)
                    fnu = np.flipud(fnu)

                    # Integrate
                    observed = splrep(nu_filter, response*fnu[nu_cond]/nu_filter, s=0, k=1)
                    flux = splint(nu_filter[0], nu_filter[-1], observed)

                    mag_grid[i,j, k] = -2.5*np.log10(flux/bandpass) -48.6 - 2.5*logmstar
    mag_dset = output_file.create_dataset('%s_mag_grid'%band, data=mag_grid)

    mag_dset.attrs['axis_0'] = 'logMstar'
    mag_dset.attrs['axis_1'] = 'Metallicity'
    mag_dset.attrs['axis_2'] = 'tau_V'
    mag_dset.attrs['axis_3'] = 'Wavelength'

output_file.create_dataset('Mstar_grid', data=Mstar_grid)
output_file.create_dataset('Z_grid', data=grid_file['Z_grid'][()])
output_file.create_dataset('tau_V_grid', data=grid_file['tau_V_grid'][()])