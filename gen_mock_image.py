from SL_simulation.mock import Deflector, Point_Source, Lense
import numpy as np
import h5py
from astropy.table import Table
from multiprocessing import Pool

lense_sample = Table.read('/Users/liurongfu/Desktop/4MOST_lensing_prediction/data/ELCOSMOS/Mock_samples.hdf5', path='lense')

#* write an id column to the table
# ids = np.arange(len(lense_sample), dtype=int)
# lense_sample.add_column(ids, name='id')
# lense_sample.write('/Users/liurongfu/Desktop/4MOST_lensing_prediction/data/ELCOSMOS/Mock_samples.hdf5', path='lense', overwrite=True, append=True)

def mock_image(id):
    print(id)
    sample = lense_sample[id]
    lense = Lense(
        z = sample['z'],
        logMh=sample['logMh'],
        logc= sample['logc'],
        gamma_DM= sample['gamma_DM'],
        logMstar= sample['logMstar'],
        alpha_sps= sample['alpha_sps'],
        Re_arcsec= sample['Re_arcsec'],
        q= sample['q'],
        radcaustic_sr= sample['radcaustic_sr'],
        num_source= sample['num_source']
    )
    lense.find_image(id)
    lense.mock_image()

    return id

args = [(id,) for id in range(len(lense_sample))]

def wrapper_mock_image(args):
    return mock_image(*args)

if __name__ == '__main__':  
    with Pool(processes=10) as pool:
        results = pool.map(wrapper_mock_image, args)
    print('Processed galaxy IDs:', results)







