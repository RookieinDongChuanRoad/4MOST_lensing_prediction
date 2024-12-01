import h5py
import numpy as np
from multiprocessing import Pool
import glafic

# file_path = '/root/4MOST_lensing_prediction/codes/data/potential_lense.h5'

def process_galaxy(id, file_path):
    import foreground_model as fg
    with h5py.File(file_path, 'r') as galaxies:
        id = str(id)
        sample = galaxies[id]

        # Extract attributes
        alpha_sps = sample.attrs['alpha_sps']
        gamma_DM = sample.attrs['gamma_DM']
        logMh = sample.attrs['logMh']
        logMstar = sample.attrs['logMstar']
        logRe = sample.attrs['logRe']
        c = 10**sample.attrs['logc']
        num_bg = sample.attrs['num_bg']
        q = sample.attrs['q']
        rhos = sample.attrs['rhos']
        rs = sample.attrs['rs']
        z_foreground = sample.attrs['z_foreground']
        logMstar_true = logMstar + np.log10(alpha_sps)
        rs = rs * fg.cosmo_astropy.arcsec_per_kpc_proper(z_foreground).value

        # Initialize glafic
        p = [
            fg.cosmo_astropy.Om0, fg.cosmo_astropy.Ode0, fg.cosmo_astropy.w(0), fg.cosmo_astropy.h,
            f'/root/4MOST_lensing_prediction/data/find_image_ELCOSMOS/{id}',
            -5, -5, 5, 5, 0.1, 0.1, 5
        ]
        glafic.init(*p)

        # Set lenses and sources
        glafic.startup_setnum(2, 0, num_bg)
        e = 1 - q

        glafic.set_lens(1, 'gnfw', z_foreground, 10**logMh * fg.cosmo_astropy.h, 0, 0, e, 90, c/(2 - gamma_DM), gamma_DM)
        glafic.set_secondary('nfw_users 0', verb=0)
        glafic.set_lens(2, 'sers', z_foreground, 10**logMstar_true * fg.cosmo_astropy.h, 0, 0, e, 90, 10**logRe * fg.cosmo_astropy.arcsec_per_kpc_proper(z_foreground).value, 4)

        for j in range(num_bg):
            z = sample['z_background'][j]
            x = sample['x_background'][j]
            y = sample['y_background'][j]
            glafic.set_point(j + 1, z, x, y)

        # Run the glafic model
        glafic.model_init()
        glafic.findimg()

    return id  # Return the galaxy ID or any result you'd like

# File path and fg object (assume `fg` is defined elsewhere in your script)
file_path = '/root/4MOST_lensing_prediction/codes/data/potential_lense_ELCOSMOS.h5'

# List of galaxy IDs to process
# galaxy_ids = list(range(1267))
# galaxy_ids = [758,
#  746,
#  745,
#  753,
#  751,
#  741,
#  756,
#  277,
#  749,
#  747,
#  279,
#  752,
#  744,
#  276,
#  740,
#  759,
#  748,
#  275,
#  743,
#  274,
#  755,
#  757,
#  754,
#  750,
#  273,
#  739,
#  278,
#  742]
galaxy_ids = [272,738]
# Example: process galaxies from ID 754 to 999

# Wrapper function to pass additional arguments
def wrapper(args):
    return process_galaxy(*args)

# Prepare arguments for each galaxy
args = [(id, file_path) for id in galaxy_ids]

# Parallel execution using Pool
if __name__ == "__main__":
    with Pool(processes=8) as pool:  # Adjust 'processes' based on the number of CPU cores
        results = pool.map(wrapper, args)

    print("Processed galaxy IDs:", results)
