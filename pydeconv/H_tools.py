import point_spread_function
from scipy.sparse import coo_matrix
import numpy as np 


def generate_H(psf):
    H = None
    pass
    return H


def generate_psf(raw_image_file):
    psf = None
    pass
    return psf


def impute_psf(raw_image_file):
    H = None
    pass

def generate_H(image,psf_function,**kwargs):
    dims = image.shape
    psf_array = psf_function(image=image,**kwargs)
    for i, psf_image in enumerate(psf_array):
        # coords = np.unravel_index(i, dims)
        # psf_image = psf_function(coords)
        # r_dist = r_map[coords]
        # sigma = sigma_scale(r_map[coords])
        # psf_image = psf.gaussian(dims=dims, mu=mu, sigma=sigma)
        # psf_window_volume[i, :, :] = psf_image
        delta_image = np.zeros_like(image)
        delta_image[np.unravel_index(i, image.shape)] = 1
        delta_PSF = scipy.ndimage.convolve(delta_image, psf_image)
        measurement_matrix[i, :] = delta_PSF.flatten()
    
    return coo_matrix(measurement_matrix)

def variable_gaussian_psf(image,psf_dims,mu,sigma_fun):
    def psf_fun(r_map):
        # psf_array = 
        sigma_map = sigma_fun(r_map)
        for i,sigma in enumerate(sigma_map):
            psf_array[coord] = point_spread_function.gaussian(psf_dims,mu,sigma)
        return psf_array
    return variable_psf(image,psf_fun)
    # image_dims = image.shape
    # grid_coords = np.meshgrid(*[np.linspace(-1, 1, image_dim) for image_dim in image_dims])
    # r_map = np.add.reduce(np.power(grid_coords,2.0))
    # sigma_map = sigma_fun(r_map)
    # r_dist = r_map[coords]
    # def sigma_scale(r_dist):
    #     return (r_dist + 0.01) * 3
    # for i,sigma in enumerate(sigma_map):
    #     coords = np.unravel_index(i, image_dims.shape)
    #     psf_array[coords] = point_spread_function.gaussian(psf_dims,mu,sigma)
    # return 

# def variable_psf(image,psf_fun):
#     image_dims = image.shape
#     grid_coords = np.meshgrid(*[np.linspace(-1, 1, image_dim) for image_dim in image_dims])
#     r_map = np.add.reduce(np.power(grid_coords,2.0))
#     psf_array = psf_fun(r_map)
#     # r_dist = r_map[coords]
#     # def sigma_scale(r_dist):
#     #     return (r_dist + 0.01) * 3
#     # for i,sigma in enumerate(sigma_map):
#     #     coords = np.unravel_index(i, image_dims.shape)
#     #     psf_array[coords] = point_spread_function.gaussian(psf_dims,mu,sigma)
#     return psf_array