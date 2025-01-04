from SL_simulation.mock import Deflector, Lense
import numpy as np
import h5py
from astropy.table import Table
from multiprocessing import Pool
import warnings
# warnings.filterwarnings("ignore")



num_deflectors = 100000
# num_density = Point_Source().fiducial_mean_N

def gen_parent(id):
    print(id)
    deflector = Deflector()
    deflector.set_self()
    attributes = vars(deflector)
    local_data = np.zeros(len(attributes) - 3)
    i = 0
    for name in attributes.keys():
        if name in ['Red', 'use_cat', 'cat']:
            continue
        local_data[i] = attributes[name]
        i += 1
    return local_data

def wrapper_gen_parent(args):
    return gen_parent(*args)

args = [(id,) for id in range(num_deflectors)]

if __name__ == '__main__':
    with Pool(processes=10) as pool:
        results = pool.map(wrapper_gen_parent, args)
    data = np.array(results).T  # Transpose to match the desired shape
    deflector = Deflector()
    deflector.set_self()
    table = Table()
    names = vars(deflector).keys()
    i = 0
    for name in names:
        if name in ['Red', 'use_cat', 'cat']:
            continue
        table[name] = data[i]
        i += 1
        
    table.write('/Users/liurongfu/Desktop/4MOST_lensing_prediction/data/ELCOSMOS/Mock_samples.hdf5', path='deflector', format='hdf5', overwrite=True,append=True)

    table_lens = table[table['num_source'] > 0]
    table_lens.write('/Users/liurongfu/Desktop/4MOST_lensing_prediction/data/ELCOSMOS/Mock_samples.hdf5', path='lense', format='hdf5', overwrite=True,append=True)




