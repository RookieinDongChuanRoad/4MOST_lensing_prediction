import numpy as np
from astropy.convolution import convolve, Moffat2DKernel
from astropy.io import fits
import h5py

#* set pixel scale
img_range = 5 # arcsec
pix_size = 0.1 # arcsec
fwhm = 0.7 # arcsec
alpha = 5 
gamma_arcsec = fwhm / (2 * np.sqrt(2**(1/alpha) - 1)) # arcsec
gamma = gamma_arcsec / pix_size
kernel = Moffat2DKernel(gamma, alpha)

#* read real lenses
with h5py.File('/root/4MOST_lensing_prediction/codes/data/real_lense_ELCOSMOS.h5', 'r+') as lenses:
    num_lenses = len(lenses)
    for i in range(num_lenses):
        #* read in image position and flux
        id = str(i)
        print(f'processing {id}-th lens \n')
        sample = lenses[id]
        oii_flux_array = sample.attrs['L_background']
        image_data = sample.attrs['image_data']
        # print(image_data)
        num_bg = int(sample.attrs['num_bg'])
        flux_in_fiber_ar = np.zeros(num_bg)
        
        image_ind = 0
        for num in range(num_bg):
            oii_flux = oii_flux_array[num]
            num_image = int(image_data[image_ind,0])
            x_img = image_data[image_ind+1:image_ind+1+num_image,0]
            y_img = image_data[image_ind+1:image_ind+1+num_image,1]
            magnification = np.abs(image_data[image_ind+1:image_ind+1+num_image,2])

            #* pixelize x,y
            x_img_pix = (img_range/pix_size + np.round(x_img/pix_size)).astype(int)
            y_img_pix = (img_range/pix_size + np.round(y_img/pix_size)).astype(int)

            #* make mock image
            mock_image = np.zeros((int(2*img_range/pix_size)+1, int(2*img_range/pix_size)+1))
            for j in range(num_image):
                mock_image[y_img_pix[j], x_img_pix[j]] = magnification[j]*oii_flux

            #* convolve with Moffat PSF
            convolved_image = convolve(mock_image, kernel)

            #* save convolved image
            x_mesh,y_mesh = np.meshgrid(np.linspace(-img_range, img_range, int(2*img_range/pix_size)+1), np.linspace(-img_range, img_range, int(2*img_range/pix_size)+1))
            r_mesh = np.sqrt(x_mesh**2 + y_mesh**2)
            fiber_size = 1.45 # arcsec
            in_fiber = r_mesh <= fiber_size
            flux_in_fiber = np.sum(convolved_image[in_fiber])
            header = fits.Header()
            header['flux_in_fiber'] = flux_in_fiber 
            header['origin_flux'] = oii_flux

            hdu1 = fits.PrimaryHDU(header=header)
            hdu2 = fits.ImageHDU(data=mock_image)
            hdu3 = fits.ImageHDU(data=convolved_image)
            hdul = fits.HDUList([hdu1, hdu2, hdu3])
            hdul.writeto(f'/root/4MOST_lensing_prediction/codes/data/mock_image_ELCOSMOS/{id}_{num}.fits', overwrite=True)
            flux_in_fiber_ar[num] = flux_in_fiber
        
            image_ind += num_image + 1
        
        sample.attrs['flux_in_fiber'] = flux_in_fiber_ar

        # image_num = int(image_data[0,0])
        # image_data = image_data[1:]
        # x_img = image_data[:,0]
        # y_img = image_data[:,1]
        # magnification = np.abs(image_data[:,2])

        # #* pixelize x,y 
        # x_img_pix = (img_range/pix_size + np.round(x_img/pix_size)).astype(int)
        # y_img_pix = (img_range/pix_size + np.round(y_img/pix_size)).astype(int)

        
        # for j in range(image_num):
        #     print(magnification[j]*oii_flux)
        #     mock_image[y_img_pix[j], x_img_pix[j]] = magnification[j]*oii_flux

        

        sample.attrs['flux_in_fiber'] = flux_in_fiber
        



