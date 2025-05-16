import sys
sys.path.append('../all_projects/')
from useful_functions import *
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from mcfost_blob_class import blob
import useful_functions as uf
plt.rcParams["image.origin"] = 'lower'
import os
import shutil
from datetime import datetime

u_coords = np.load('u_coords.npy')
v_coords = np.load('v_coords.npy')

wavelength = 750 * 10 ** (-9)
pixel_ratio = 1

ucoords = np.expand_dims(u_coords, axis = 1)
vcoords = np.expand_dims(v_coords, axis = 1)
uv_coords = np.concatenate((ucoords, vcoords), axis = 1)
x, y, z = np.ogrid[-50: 51, -50: 51, -50: 51]
xx, yy = np.meshgrid(x.flatten(), y.flatten())
dftm_grid = uf.compute_DFTM1(xx.flatten(), yy.flatten(), uv_coords, wavelength)

bl = np.sqrt(u_coords ** 2 + v_coords ** 2)
az = np.arctan(v_coords/u_coords)
indx_of_cp = np.load('indx_of_cp.npy')

data_dir = 'path_to_your_data'
dataset_number = 'dataset_1'
os.mkdir(data_dir + dataset_number)



unique_name = 'test_dataset'
date_current = datetime.now()
dayy = date_current.day
monthh = date_current.month
yearr = date_current.year

if dayy < 10:
    dayy = str(0) + str(dayy)
if monthh < 10:
    monthh = str(0) + str(monthh)


month_day_year = str(dayy) + str(monthh) + str(yearr)

image_size = 101
num_images = 100
num_batch = 100
num_stokes = 4
example_num = int(31e3)
image_size_pix = image_size
grid_size_AU = 10


for batch in range(0, num_batch):

    image_params_store = {}
    stokes_images = np.zeros((num_images, num_stokes, image_size, image_size), dtype=np.float32)
    observables = np.zeros((num_images, num_stokes, 153 + 816), dtype=np.float32)
    cart_grids = np.zeros((image_size, image_size))

    for i in range(0, num_images):

        seed = example_num
        np.random.seed(seed)

        numberofblob = np.random.randint(10, 25)
        model_params = {'seed': example_num,
                        'image_size': image_size_pix,
                        'mcfost_diag': False,
                        'pixel_ratio': pixel_ratio,
                        'wavelength': wavelength,
                        'dftm_grid': dftm_grid,
                        'distance': '100',
                        'grid_nx': image_size_pix,
                        'grid_ny': image_size_pix,
                        'size': grid_size_AU,
                        'star_radius_pix': np.random.uniform(5,15),
                        'n_rad': '100',
                        'n_z': '30',
                        'n_az': '70',
                        'indx_of_cp': indx_of_cp,
                        'zdata': bl,
                        'xdata': az,
                        'number_of_blobs': numberofblob,
                        'dust_density':   np.random.uniform(1e-25, 5e-15, numberofblob),
                        'amin': '0.001',  # 1 nm
                        'amax': '0.300',  # 300 nm
                        'n_grains': '100',
                        'nbr_photons': '1.3e7'}


        mcfost_model = blob(model_params)
        mcfost_model.make_blob_density_file()

        cart_grid = mcfost_model.cartesian_grid.sum(axis=2)  # can double check this summing direction
        cart_grids = cart_grids + cart_grid  # /cart_grid.sum()
        mcfost_model.run_mcfost()

        mcfost_model.make_pol_diff_vis()

        if model_params['mcfost_diag'] == True:
            uf.plot_model(mcfost_model, i, i)
            uf.plot_mcfost_model_diags(mcfost_model.cartesian_grid.sum(axis=2), mcfost_model, i,i)

        mcfost_model.params_as_dict()
        image_params_store[i] = mcfost_model.params_out

        stokes_images[i,] = mcfost_model.final_images
        observables[i,] = np.hstack((mcfost_model.final_vis, mcfost_model.final_cp))

        example_num = example_num + 1

    np.savez(data_dir + dataset_number + '/blob_data_{}_{}_{}.npz'.format(folder_number, unique_name, month_day_year, batch),
             stokes_images=stokes_images, observables=observables, parameters=image_params_store)


    plt.figure()
    plt.imshow(cart_grids)
    plt.colorbar()
    plt.savefig( data_dir + dataset_number '/example_dist_{}.pdf'.format( folder_number, batch))


