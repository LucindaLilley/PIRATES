import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
import pickle
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dense
import psutil
import os
import subprocess
import sys
from scipy.ndimage import zoom

from skimage import img_as_float
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tensorflow.keras.utils import plot_model
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras import backend as K
from keras.utils import custom_object_scope
 


def normalize_image(img):
    """Normalize image to the range [0, 1]."""
    img_min = np.min(img)
    img_max = np.max(img)
    return (img - img_min) / (img_max - img_min)


def pol_ssim(unnorm_pred_it_stokes, unnorm_real_stokes):

    lowest_min = np.min((unnorm_pred_it_stokes[0, :, :, 0], unnorm_real_stokes[0, :, :, 0]))
    highest_max = np.max((unnorm_pred_it_stokes[0, :, :, 0], unnorm_real_stokes[0, :, :, 0]))
    current_range = highest_max - lowest_min

    I_pl, c = ssim(unnorm_pred_it_stokes[0, :, :, 0], unnorm_real_stokes[0, :, :, 0], full=True,data_range=current_range)

    lowest_min = np.min((unnorm_pred_it_stokes[0, :, :, 1], unnorm_real_stokes[0, :, :, 1]))
    highest_max = np.max((unnorm_pred_it_stokes[0, :, :, 1], unnorm_real_stokes[0, :, :, 1]))
    current_range = highest_max - lowest_min

    Q_pl, c  = ssim(unnorm_pred_it_stokes[0, :, :, 1], unnorm_real_stokes[0, :, :, 1], full=True, data_range=current_range)

    lowest_min = np.min((unnorm_pred_it_stokes[0, :, :, 2], unnorm_real_stokes[0, :, :, 2]))
    highest_max = np.max((unnorm_pred_it_stokes[0, :, :, 2], unnorm_real_stokes[0, :, :, 2]))
    current_range = highest_max - lowest_min

    U_pl, c  = ssim(unnorm_pred_it_stokes[0, :, :, 2], unnorm_real_stokes[0, :, :, 2], full=True, data_range=current_range)

    im_pl = (I_pl + Q_pl + U_pl)*(1/3)

    return im_pl





@keras.saving.register_keras_serializable()
def custom_loss_with_arguments(dftm_grid, indx_of_cp, normfacts_y, real_obs_norm, true_err, normfacts_X, reg_strength,  fisher_dict, old_params_dict, model):
    @keras.saving.register_keras_serializable()
    def custom_loss_function_tensor(y_true, y_pred):


        y_pred = y_pred[0, :, :, :]
        zero_pad = tf.zeros([128, 128, 1])
        y_pred = tf.concat((y_pred, zero_pad), axis=2)
        y_pred = unnorm_images_tensor(y_pred, normfacts_y)
        y_pred = out_of_network(y_pred)



        pred_I_vis, pred_I_cp, pred_Q_vis, pred_Q_cp, pred_U_vis, pred_U_cp  = stokes_2_vampires_bispect_tensor(y_pred, dftm_grid, indx_of_cp, 0, 0, 0)

        pred_Q_cp = pred_Q_cp * (np.pi / 180)
        pred_U_cp = pred_U_cp * (np.pi / 180)
        pred_I_cp = pred_I_cp * (np.pi / 180)

        pred_I_vis, pred_I_cp, pred_Q_vis, pred_Q_cp, pred_U_vis, pred_U_cp = norm_observables_tensor(pred_I_vis, pred_I_cp, pred_Q_vis, pred_Q_cp, pred_U_vis, pred_U_cp, normfacts_X)


        true_things =           real_obs_norm
        predicted_things =      tf.concat([pred_I_vis, pred_I_cp, pred_Q_vis, pred_Q_cp, pred_U_vis, pred_U_cp], axis=0) #

        difference = tf.subtract(true_things, predicted_things)
        squared_difference = tf.square(difference)
        squared_diff_err = squared_difference/true_err
        rex = tf.reduce_mean(squared_diff_err, axis=-1)

        ewc_loss = 0.0
        for var in model.trainable_variables:
            ref = var.ref()
            if ref in fisher_dict:
                fisher = fisher_dict[ref]
                old_param = old_params_dict[ref]
                param_diff = var - old_param
                ewc_loss += tf.reduce_sum(fisher * tf.square(param_diff))
        ewc_loss *= reg_strength

        return rex + ewc_loss

    return custom_loss_function_tensor


def study_through_lp_tensor(final_stokes):

        lp_H = 0.5 * tf.constant([[1, 1, 0, 0],
                                  [1, 1, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]], dtype=tf.float32)
        lp_V = 0.5 * tf.constant([[1, -1, 0, 0],
                                  [-1, 1, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]], dtype=tf.float32)
        lp_H45 = 0.5 * tf.constant([[1, 0, 1, 0],
                                    [0, 0, 0, 0],
                                    [1, 0, 1, 0],
                                    [0, 0, 0, 0]], dtype=tf.float32)
        lp_V45 = 0.5 * tf.constant([[1, 0, -1, 0],
                                    [0, 0, 0, 0],
                                    [-1, 0, 1, 0],
                                    [0, 0, 0, 0]], dtype=tf.float32)
        # 101, 101, 4
        channel_0 = tf.reshape(final_stokes[:, :, 0:1], [1, 128, 128])
        channel_1 = tf.reshape(final_stokes[:, :, 1:2], [1, 128, 128])
        channel_2 = tf.reshape(final_stokes[:, :, 2:3], [1, 128, 128])
        channel_3 = tf.zeros(shape=(1, 128, 128))


        # Concatenate the channels back together
        final_stokes = tf.concat([channel_0, channel_1, channel_2, channel_3], axis=0)

        final_stokes_flat = tf.reshape(final_stokes, (4, 128 * 128))
        H = tf.matmul(lp_H, final_stokes_flat)
        H = tf.reshape(H, [4, 128, 128])
        V = tf.matmul(lp_V, final_stokes_flat)
        V = tf.reshape(V, [4, 128, 128])
        H45 = tf.matmul(lp_H45, final_stokes_flat)
        H45 = tf.reshape(H45, [4, 128, 128])
        V45 = tf.matmul(lp_V45, final_stokes_flat)
        V45 = tf.reshape(V45, [4, 128, 128])
        H = H[0, :, :]
        V = V[0, :, :]
        H45 = H45[0, :, :]
        V45 = V45[0, :, :]
        return H, V, H45, V45

def calc_vis_tensor(fft):
    vis = tf.square(tf.abs(fft))
    return vis


def apply_DFTM1_tensor(image, dftm):
    '''Apply a direct Fourier transform matrix to an image.'''
    image /= tf.keras.backend.sum(image)
    a = tf.convert_to_tensor(dftm, dtype = tf.complex128)
    b = tf.cast(image, tf.float64)
    zeros_e = tf.cast(tf.zeros_like(image), tf.float64)
    b = tf.complex(b, zeros_e)
    b = tf.keras.backend.flatten(b)
    b = tf.expand_dims(b, 1)
    return tf.keras.backend.dot(a, b)


def calc_cp_tensor(fft_arr, indx_of_cp):
    indx_ = tf.constant(indx_of_cp)
    cvis_cals = tf.gather(fft_arr, indx_) 
    bispectrum = cvis_cals[0, :, 0] * cvis_cals[1, :, 0] * tf.math.conj(cvis_cals[2, :, 0])
    closure_phases = tf.math.angle(bispectrum)
    closure_phases = closure_phases * (180/np.pi)
    return closure_phases


def norm_observables_tensor(I_vis, I_cp,  Q_vis,  Q_cp,  U_vis,  U_cp, normfacts_X):

    I_vis = (I_vis - normfacts_X['mean_I_vis'])/(normfacts_X['sd_I_vis'])
    Q_vis = (Q_vis - normfacts_X['mean_Q_vis'])/(normfacts_X[ 'sd_Q_vis'])
    U_vis = (U_vis - normfacts_X['mean_U_vis'])/(normfacts_X['sd_U_vis'])

    I_cp = (I_cp - normfacts_X['mean_I_cp'])/(normfacts_X['sd_I_cp'])
    Q_cp = (Q_cp - normfacts_X['mean_Q_cp'])/(normfacts_X['sd_Q_cp'])
    U_cp = (U_cp - normfacts_X['mean_U_cp'])/(normfacts_X['sd_U_cp'])

    return I_vis, I_cp, Q_vis, Q_cp, U_vis,  U_cp

def stokes_2_vampires_bispect_tensor(stokes, dftm_grid, indx_of_cp, bl, az, plot):


    stokes = tf.cast(stokes, dtype=tf.float32)

    H, V, H45, V45 = study_through_lp_tensor(stokes)
    Itemp = H + V
    Hvis = calc_vis_tensor(apply_DFTM1_tensor(H, dftm_grid))
    Vvis = calc_vis_tensor(apply_DFTM1_tensor(V, dftm_grid))
    H45vis = calc_vis_tensor(apply_DFTM1_tensor(H45, dftm_grid))
    V45vis = calc_vis_tensor(apply_DFTM1_tensor(V45, dftm_grid))

    true_I_vis =     tf.cast(calc_vis_tensor(apply_DFTM1_tensor(Itemp, dftm_grid)), tf.float32)
    true_Q_vis =     tf.cast(Vvis / Hvis, tf.float32)
    true_U_vis =     tf.cast(V45vis / H45vis, tf.float32)


    Hcp = calc_cp_tensor(apply_DFTM1_tensor(H, dftm_grid), indx_of_cp)
    Vcp = calc_cp_tensor(apply_DFTM1_tensor(V, dftm_grid), indx_of_cp)
    H45cp = calc_cp_tensor(apply_DFTM1_tensor(H45, dftm_grid), indx_of_cp)
    V45cp = calc_cp_tensor(apply_DFTM1_tensor(V45, dftm_grid), indx_of_cp)
    true_Q_cp = Vcp - Hcp
    true_U_cp = V45cp - H45cp
    true_I_cp = calc_cp_tensor(apply_DFTM1_tensor(Itemp, dftm_grid), indx_of_cp)


    true_I_cp = tf.cast(true_I_cp, tf.float32)
    true_I_vis = tf.cast(true_I_vis, tf.float32)[:,0]

    true_Q_vis = tf.cast(true_Q_vis, tf.float32)[:,0]
    true_U_vis = tf.cast(true_U_vis, tf.float32)[:,0]

    true_Q_cp = tf.cast(true_Q_cp, tf.float32)
    true_U_cp = tf.cast(true_U_cp, tf.float32)



    return true_I_vis, true_I_cp, true_Q_vis, true_Q_cp, true_U_vis, true_U_cp



def out_of_network(stokes):

    channel_0 = tf.expand_dims(stokes[:, :, 0], axis=2)
    channel_1 = tf.expand_dims(stokes[:, :, 1], axis=2)
    channel_2 = tf.expand_dims(stokes[:, :, 2], axis=2)
    channel_3 = tf.expand_dims(stokes[:, :, 3], axis=2)

    channel_0 = 10 ** channel_0

    y_true = tf.concat([channel_0, channel_1, channel_2, channel_3], axis=len(tf.shape(stokes)) - 1)

    return y_true

def unnorm_images_tensor(stokes, normfacts_y):


    dim1 = tf.shape(stokes)[0]
    dim2 = tf.shape(stokes)[1]
    dim3 = tf.shape(stokes)[2]

    channel_0 = tf.expand_dims(stokes[:, :, 0], axis=2)
    channel_1 = tf.expand_dims(stokes[:, :, 1], axis=2)
    channel_2 = tf.expand_dims(stokes[:, :, 2], axis=2)
    channel_3 = tf.expand_dims(stokes[:, :, 3], axis=2)

    channel_0 = (channel_0 * normfacts_y['sd_I_image']) + normfacts_y['mean_I_image']
    channel_1 = (channel_1 * normfacts_y['sd_Q_image']) + normfacts_y['mean_Q_image']
    channel_2 = (channel_2 * normfacts_y['sd_U_image']) + normfacts_y['mean_U_image']
    channel_3 = channel_3

    y_true = tf.concat([channel_0, channel_1, channel_2, channel_3], axis=2)

    return y_true


@register_keras_serializable()
def custom_image_with_arguments():
    @register_keras_serializable()
    def custom_images(y_true, y_pred):
        differencesq2 = tf.keras.backend.square(y_true - y_pred)
        rex2 = tf.keras.backend.mean(differencesq2)

        return rex2
    return custom_images


def gkern(l=5, sig=1.):

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)



def compute_DFTM1(x, y, uv, wavel):
    '''Compute a direct Fourier transform matrix, from coordinates x and y (milliarcsec) to uv (metres) at a given wavelength wavel.'''

    # Convert to radians
    x = x * np.pi / 180.0 / 3600.0 / 1000.0
    y = y * np.pi / 180.0 / 3600.0 / 1000.0

    # get uv in nondimensional units
    uv = uv / wavel
    # Compute the matrix
    dftm = np.exp(-2j * np.pi * (np.outer(uv[:, 0], x) + np.outer(uv[:, 1], y)))
    return dftm


def unnorm_images(stokes, norm_params):
    if len(np.shape(stokes)) == 3:
        dim1 = np.shape(stokes)[0]
        dim2 = np.shape(stokes)[1]
        dim3 = np.shape(stokes)[2]

        if dim1 == 128 and dim2 == 128:
            stokes[:, :, 0] = (stokes[:, :, 0] * norm_params['sd_I_image']) + norm_params['mean_I_image']
            stokes[:, :, 1] = (stokes[:, :, 1] * norm_params['sd_Q_image']) + norm_params['mean_Q_image']
            stokes[:, :, 2] = (stokes[:, :, 2] * norm_params['sd_U_image']) + norm_params['mean_U_image']
            stokes[:, :, 3] = stokes[:, :, 3]

        else:

            stokes[0, :, :] = (stokes[0, :, :] * norm_params['sd_I_image']) + norm_params['mean_I_image']
            stokes[1, :, :] = (stokes[1, :, :] * norm_params['sd_Q_image']) + norm_params['mean_Q_image']
            stokes[2, :, :] = (stokes[2, :, :] * norm_params['sd_U_image']) + norm_params['mean_U_image']
            stokes[3, :, :] = stokes[3, :, :]

    elif len(np.shape(stokes)) == 4:

        dim1 = np.shape(stokes)[0]
        dim2 = np.shape(stokes)[1]
        dim3 = np.shape(stokes)[2]
        dim4 = np.shape(stokes)[3]

        if dim4 == 4:

            stokes[:, :, :, 0] = (stokes[:, :, :, 0] * norm_params['sd_I_image']) + norm_params['mean_I_image']
            stokes[:, :, :, 1] = (stokes[:, :, :, 1] * norm_params['sd_Q_image']) + norm_params['mean_Q_image']
            stokes[:, :, :, 2] = (stokes[:, :, :, 2] * norm_params['sd_U_image']) + norm_params['mean_U_image']
            stokes[:, :, :, 3] = stokes[:, :, :, 3]

        elif dim1 == 4:

            stokes[0, :, :, :] = (stokes[0, :, :, :] * norm_params['sd_I_image']) + norm_params['mean_I_image']
            stokes[1, :, :, :] = (stokes[1, :, :, :] * norm_params['sd_Q_image']) + norm_params['mean_Q_image']
            stokes[2, :, :, :] = (stokes[2, :, :, :] * norm_params['sd_U_image']) + norm_params['mean_U_image']
            stokes[3, :, :, :] = stokes[3, :, :, :]

    return stokes



def study_through_lp(final_stokes):  

    im_size = np.shape(final_stokes)[1]
    lp_H = 0.5 * np.array([[1, 1, 0, 0],
                           [1, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])

    lp_V = 0.5 * np.array([[1, -1, 0, 0],
                           [-1, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])

    lp_H45 = 0.5 * np.array([[1, 0, 1, 0],
                             [0, 0, 0, 0],
                             [1, 0, 1, 0],
                             [0, 0, 0, 0]])

    lp_V45 = 0.5 * np.array([[1, 0, -1, 0],
                             [0, 0, 0, 0],
                             [-1, 0, 1, 0],
                             [0, 0, 0, 0]])

    if np.shape(final_stokes)[-1] == 4:
        final_I = np.expand_dims(final_stokes[:, :, 0], axis=0)
        final_Q = np.expand_dims(final_stokes[:, :, 1], axis=0)
        final_U = np.expand_dims(final_stokes[:, :, 2], axis=0)
        final_V = np.expand_dims(final_stokes[:, :, 3], axis=0)

        final_stokes = np.concatenate((final_I, final_Q, final_U, final_V), axis=0)

    final_stokes_flat = np.reshape(final_stokes, (4, im_size * im_size))  

    H = lp_H @ final_stokes_flat  
    H = np.reshape(H, (4, im_size, im_size))
    V = lp_V @ final_stokes_flat
    V = np.reshape(V, (4, im_size, im_size))
    H45 = lp_H45 @ final_stokes_flat
    H45 = np.reshape(H45, (4, im_size, im_size))
    V45 = lp_V45 @ final_stokes_flat
    V45 = np.reshape(V45, (4, im_size, im_size))
    H = H[0, :, :]
    V = V[0, :, :]
    H45 = H45[0, :, :]
    V45 = V45[0, :, :]
    return H, V, H45, V45

def window_amical(datacube, window, m = 3):

    cleaned_array = np.zeros(np.shape(datacube))

    for i in range(np.shape(datacube)[0]):

        img = datacube[i,:,:]
        isz = len(img)
        xx, yy = np.arange(isz), np.arange(isz)
        xx2 = (xx-isz//2)
        yy2 = (isz//2-yy)

        distance = np.sqrt(xx2**2 + yy2[:, np.newaxis]**2)

        if np.shape(window*2) != ():
            tep = 3
        w = super_gaussian(distance, sigma=window*2, m=m)
        img_apod = img * w
        cleaned_array[i,:,:] = img_apod

    return cleaned_array, w

def calc_vis(fft):
    vis = np.abs(fft) ** 2
    return vis


def calc_cp(fft_arr, indx_of_cp):
    cvis_cals = fft_arr[indx_of_cp]  
    bispectrum = cvis_cals[0,:] * cvis_cals[1,:] * np.conj(cvis_cals[2,:])
    closure_phases = np.angle(bispectrum, deg = True) 
    return closure_phases


def apply_DFTM1(image, dftm): 
    image /= image.sum()  
    return np.dot(dftm, image.ravel())



def super_gaussian(x, sigma, m, amp=1, x0=0):
    sigma = float(sigma)
    m = float(m)

    return amp * ((np.exp(-(2 ** (2 * m - 1)) * np.log(2) * (((x - x0) ** 2) / ((sigma) ** 2)) ** (m))) ** 2)



def stokes_2_vampires_bispect(stokes, dftm_grid, indx_of_cp, bl, az ):
    
    H, V, H45, V45 = study_through_lp(stokes)
    Itemp = H + V
    Hvis = calc_vis(apply_DFTM1(H, dftm_grid))
    Vvis = calc_vis(apply_DFTM1(V, dftm_grid))
    H45vis = calc_vis(apply_DFTM1(H45, dftm_grid))
    V45vis = calc_vis(apply_DFTM1(V45, dftm_grid))
    true_I_vis = calc_vis(apply_DFTM1(Itemp, dftm_grid))
    true_Q_vis = Vvis / Hvis
    true_U_vis = V45vis / H45vis

    Hcp = calc_cp(apply_DFTM1(H, dftm_grid), indx_of_cp)
    Vcp = calc_cp(apply_DFTM1(V, dftm_grid), indx_of_cp)
    H45cp = calc_cp(apply_DFTM1(H45, dftm_grid), indx_of_cp)
    V45cp = calc_cp(apply_DFTM1(V45, dftm_grid), indx_of_cp)
    true_I_cp = calc_cp(apply_DFTM1(Itemp, dftm_grid), indx_of_cp)
    true_Q_cp = Vcp - Hcp
    true_U_cp = V45cp - H45cp

  

    return true_I_vis, true_I_cp, true_Q_vis, true_Q_cp, true_U_vis, true_U_cp


def stokes_2_vampires_bispect_noise(stokes, dftm_grid, indx_of_cp, bl, az ):

    H, V, H45, V45 = study_through_lp(stokes)
    Itemp = H + V

    Hvis = calc_vis(apply_DFTM1(H, dftm_grid))
    Vvis = calc_vis(apply_DFTM1(V, dftm_grid))
    H45vis = calc_vis(apply_DFTM1(H45, dftm_grid))
    V45vis = calc_vis(apply_DFTM1(V45, dftm_grid))
    true_I_vis = calc_vis(apply_DFTM1(Itemp, dftm_grid))


    Hcp = calc_cp(apply_DFTM1(H, dftm_grid), indx_of_cp)
    Vcp = calc_cp(apply_DFTM1(V, dftm_grid), indx_of_cp)
    H45cp = calc_cp(apply_DFTM1(H45, dftm_grid), indx_of_cp)
    V45cp = calc_cp(apply_DFTM1(V45, dftm_grid), indx_of_cp)
    true_I_cp = calc_cp(apply_DFTM1(Itemp, dftm_grid), indx_of_cp)

    true_Q_vis = Vvis / Hvis
    true_U_cp = V45cp - H45cp
    true_U_vis = V45vis / H45vis
    true_Q_cp = Vcp - Hcp

    true_U_cp = np.deg2rad(true_U_cp)
    true_Q_cp = np.deg2rad(true_Q_cp)
    true_I_cp = np.deg2rad(true_I_cp)


    #############################

    frac_noise = 1/25
    means = true_I_vis
    std_devs = frac_noise*np.mean(true_I_vis)*np.ones(np.shape(true_I_vis))
    true_I_vis = np.random.normal(loc=means, scale=std_devs, size=153)
    true_I_vis_err = std_devs


    frac_noise = 1/12
    means = true_Q_vis
    std_devs = frac_noise*np.mean(np.abs((1-true_Q_vis))) * np.ones(np.shape(true_Q_vis))
    true_Q_vis = np.random.normal(loc=means, scale=std_devs, size=153)
    true_Q_vis_err = std_devs

    frac_noise = 1/12
    means = true_U_vis
    std_devs = frac_noise*np.mean(np.abs((1-true_U_vis))) * np.ones(np.shape(true_U_vis))
    true_U_vis = np.random.normal(loc=means, scale=std_devs, size=153)
    true_U_vis_err = std_devs


    #############################

    frac_noise = 1/3
    means = true_I_cp
    std_devs = np.mean(np.abs(frac_noise*true_I_cp))*np.ones(np.shape(true_I_cp))
    true_I_cp = np.random.normal(loc=means, scale=std_devs, size=816)
    true_I_cp_err = std_devs

    frac_noise = 1/3
    means = true_Q_cp
    std_devs =  np.mean(np.abs(frac_noise*true_Q_cp))*np.ones(np.shape(true_Q_cp))
    true_Q_cp = np.random.normal(loc=means, scale=std_devs, size=816)
    true_Q_cp_err = std_devs

    frac_noise = 1/3
    means = true_U_cp
    std_devs = np.mean(np.abs(frac_noise*true_U_cp))*np.ones(np.shape(true_U_cp))
    true_U_cp = np.random.normal(loc=means, scale=std_devs, size=816)
    true_U_cp_err = std_devs


    true_obs =     np.concatenate((true_I_vis,        true_I_cp,       true_Q_vis,     true_Q_cp,     true_U_vis,      true_U_cp), axis = 0)
    true_obs_err = np.concatenate((true_I_vis_err, true_I_cp_err, true_Q_vis_err, true_Q_cp_err, true_U_vis_err,  true_U_cp_err), axis = 0)

    return true_obs, true_obs_err



class imrnn:
    def __init__(self, datadir, model_name):

        self.datadir = datadir
        self.model_name = model_name + '/'

    def load_training_data(self, data_dir):

 
        self.u_coords = np.load(self.datadir + '/u_coords.npy')
        self.v_coords = np.load(self.datadir + '/v_coords.npy')

        ucoords = np.expand_dims(self.u_coords, axis=1)
        vcoords = np.expand_dims(self.v_coords, axis=1)
        uv_coords = np.concatenate((ucoords, vcoords), axis=1)


        self.wavelength = 750 * 10 ** (-9)
        self.bl = np.sqrt(self.u_coords ** 2 + self.v_coords ** 2)
        self.az = np.arctan(self.v_coords / self.u_coords)
 
        self.indx_of_cp = np.load(self.datadir +  '/indx_of_cp.npy')

        self.ydata = np.zeros((1, 3, 128, 128))
        self.Xdata = np.zeros((1, 2907))
        self.stellar_radii = np.zeros((1,))

        x, y, z = np.ogrid[-128:128:2, -128:128:2, -128:128:2]
        xx, yy = np.meshgrid(x.flatten(), y.flatten())
        self.dftm_grid = compute_DFTM1(xx.flatten(), yy.flatten(), uv_coords, self.wavelength)


        self.ydata = np.load(self.datadir + self.model_name + 'pre-saved_y.npy')
        self.Xdata = np.load(self.datadir + self.model_name + 'pre_saved_x.npy')

        self.ydata = self.ydata[1:, ]
        self.Xdata = self.Xdata[1:, ]

        self.ydata[:, 0, ] = np.log10(self.ydata[:, 0, ])

        mn_Iim = np.mean(self.ydata[:, 0, :, :])
        mn_Qim = np.mean(self.ydata[:, 1, :, :])
        mn_Uim = np.mean(self.ydata[:, 2, :, :])

        sd_Iim = np.std(self.ydata[:, 0, :, :])
        sd_Qim = np.std(self.ydata[:, 1, :, :])
        sd_Uim = np.std(self.ydata[:, 2, :, :])

        self.ydata[:, 0, :, :] = (self.ydata[:, 0, :, :] - mn_Iim) / (sd_Iim)
        self.ydata[:, 1, :, :] = (self.ydata[:, 1, :, :] - mn_Qim) / (sd_Qim)
        self.ydata[:, 2, :, :] = (self.ydata[:, 2, :, :] - mn_Uim) / (sd_Uim)


        self.normfacts_y = {'mean_I_image': mn_Iim, 'mean_Q_image': mn_Qim, 'mean_U_image': mn_Uim,
                            'sd_I_image': sd_Iim, 'sd_Q_image': sd_Qim, 'sd_U_image': sd_Uim}

        with open(self.datadir + self.model_name + 'normfacts_y', 'wb') as fp:
            pickle.dump(self.normfacts_y, fp)


        mn_Ivis = np.mean(self.Xdata[:, 0:153])      # 153
        mn_Icp = np.mean(self.Xdata[:, 153:969])     # 816

        mn_Qvis = np.mean(self.Xdata[:,  969:1122])  # 153
        mn_Qcp = np.mean(self.Xdata[:,   1122:1938]) # 816

        mn_Uvis = np.mean(self.Xdata[:,  1938:2091]) # 153
        mn_Ucp = np.mean(self.Xdata[:,   2091:2907]) # 816


        std_Ivis = np.std(self.Xdata[:, 0:153])      # 153
        std_Icp = np.std(self.Xdata[:, 153:969])     # 816

        std_Qvis = np.std(self.Xdata[:,  969:1122])  # 153
        std_Qcp = np.std(self.Xdata[:,   1122:1938]) # 816

        std_Uvis = np.std(self.Xdata[:,  1938:2091]) # 153
        std_Ucp = np.std(self.Xdata[:,   2091:2907]) # 816


        self.Xdata[:, 0:153] =      (self.Xdata[:, 0:153]     - mn_Ivis)/(std_Ivis)
        self.Xdata[:, 153:969] =    (self.Xdata[:, 153:969]   - mn_Icp)/(std_Icp)

        self.Xdata[:, 969:1122] =   (self.Xdata[:, 969:1122]  - mn_Qvis)/(std_Qvis)
        self.Xdata[:, 1122:1938] =  (self.Xdata[:, 1122:1938] - mn_Qcp)/(std_Qcp)

        self.Xdata[:, 1938:2091] =  (self.Xdata[:, 1938:2091] - mn_Uvis)/(std_Uvis)
        self.Xdata[:, 2091:2907] =  (self.Xdata[:, 2091:2907] - mn_Ucp)/(std_Ucp)


        plt.figure(figsize=(15, 4))
        plt.subplot(1, 6, 1)
        plt.hist(self.Xdata[:, 0:153].flatten(), bins=100)
        plt.title('I vis')
        plt.subplot(1, 6, 2)
        plt.hist( self.Xdata[:, 153:969].flatten(), bins=100)
        plt.title('I cp')
        plt.subplot(1, 6, 3)
        plt.hist(self.Xdata[:, 969:1122].flatten(), bins=100)
        plt.title('Q vis')
        plt.subplot(1, 6, 4)
        plt.hist(self.Xdata[:, 1122:1938].flatten(), bins=100)
        plt.title('Q cp')
        plt.subplot(1, 6, 5)
        plt.hist(self.Xdata[:, 1938:2091].flatten(), bins=100)
        plt.title('U vis')
        plt.subplot(1, 6, 6)
        plt.hist(self.Xdata[:, 2091:2907].flatten(), bins=100)
        plt.title('U cp')
        plt.tight_layout()
        plt.savefig(self.datadir + self.model_name + 'savefigs/' + 'input_distributions.pdf')
        plt.close('all')


        self.normfacts_X = {'mean_I_vis': mn_Ivis,
                            'mean_Q_vis': mn_Qvis,
                            'mean_U_vis': mn_Uvis,

                            'sd_I_vis': std_Ivis,
                            'sd_Q_vis': std_Qvis,
                            'sd_U_vis': std_Uvis,

                            'mean_I_cp': mn_Icp,
                            'mean_Q_cp': mn_Qcp,
                            'mean_U_cp': mn_Ucp,

                            'sd_I_cp': std_Icp,
                            'sd_Q_cp': std_Qcp,
                            'sd_U_cp': std_Ucp}

        with open(self.datadir + self.model_name + 'normfacts_X', 'wb') as fp:
            pickle.dump(self.normfacts_X, fp)
 
        for i in range(1,3):
            print(i)
            self.ydata = np.swapaxes(self.ydata, i, i + 1)


        plt.close('all')
        print(np.shape(self.Xdata))
        print(np.shape(self.ydata))
        np.random.seed(42)

        ind_sample = np.random.randint(0, np.shape(self.Xdata)[0], size=self.pdict['num_test'])
        all_integers = np.arange(0, np.shape(self.Xdata)[0])
        not_included_integers = np.setdiff1d(all_integers, ind_sample)

        print('The number of training egs is {} ************'.format(self.pdict['num_train']))
        print('The number of validation egs is {} *************'.format(self.pdict['num_test']))

        not_included_subset = np.random.choice(not_included_integers, self.pdict['num_train'], replace=False)

        print('The sum of the ind samples is ............. {}'.format(np.sum(ind_sample)))


        self.X_test = self.Xdata[ind_sample,]
        self.y_test = self.ydata[ind_sample,]

        self.X_train = self.Xdata[not_included_subset,]
        self.y_train = self.ydata[not_included_subset,]

        print(np.shape(self.X_test))
        print(np.shape(self.y_test))

        print(np.shape(self.X_train))
        print(np.shape(self.y_train))



    def build_model_transpose_layers(self):


        Xndims = self.Xdata.shape[1]
        yndims = self.ydata.shape[-1]

        k_sz = self.pdict['kernel_size']
        n_lay = self.pdict['n_lay']
        num_kern = self.pdict['num_kern']
        num_kern_first = self.pdict['num_kern_first']

        reshape_spatial = self.pdict['reshape_spatial']
        reshape_channel = self.pdict['reshape_channel']
        num_stride2 = self.pdict['stride_layers']
        leakrel = self.pdict['leaky_relu']

        model = models.Sequential()
        model.add(layers.Dense(4096, input_shape=(Xndims,) ))
        model.add(LeakyReLU(alpha=leakrel)) # possibly unneeded
        model.add(layers.Reshape((reshape_spatial, reshape_spatial, reshape_channel)))

        for i in range(num_stride2):

            model.add(layers.Conv2DTranspose(filters=int(np.ceil(num_kern_first)) , kernel_size=(k_sz, k_sz), strides=2, padding='same'))
            model.add(LeakyReLU(alpha=leakrel))
            model.add(layers.Dropout(self.pdict['dropout_rate']))



        for i in range(n_lay):

            model.add(layers.Conv2DTranspose(filters=int(np.ceil(num_kern)) , kernel_size=(k_sz, k_sz), strides=1, padding='same'))
            model.add(LeakyReLU(alpha=leakrel))
            model.add(layers.Dropout(self.pdict['dropout_rate']))


        model.add(layers.Conv2DTranspose(filters=int(np.ceil(8)) , kernel_size=(k_sz, k_sz), strides=1, padding='same'))
        model.add(LeakyReLU(alpha=leakrel))
        model.add(layers.Dropout(self.pdict['dropout_rate']))


        model.add(layers.Conv2DTranspose(filters=yndims, kernel_size=(k_sz, k_sz), strides=1, padding='same', activation='linear'))
        model.summary()

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.pdict['learningRate']), loss = custom_image_with_arguments()) #\


        self.model = model

        return model


    def train_model(self, pdict, testdatasplit=0.2, num_preds=15, showplots=False, tag = ''):

        self.pdict = pdict


        class StopWhenLRBelowThreshold(Callback):
            def __init__(self, lr_threshold, check_interval=10):
                super(StopWhenLRBelowThreshold, self).__init__()
                self.lr_threshold = lr_threshold
                self.check_interval = check_interval

            def on_epoch_end(self, epoch, logs=None):
                # Only check the learning rate every 'check_interval' epochs
                if (epoch + 1) % self.check_interval == 0:
                    current_lr = self.model.optimizer.lr.read_value()
                    if current_lr <= self.lr_threshold:
                        print(
                            f"\nEpoch {epoch + 1}: Stopping training as learning rate {current_lr} is below the threshold of {self.lr_threshold}.")
                        self.model.stop_training = True
                    else:
                        print(f"\nEpoch {epoch + 1}: Learning rate is {current_lr}, continuing training.")

        # Existing ReduceLROnPlateau callback
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.8,
            patience=5,
            verbose=1,
            mode='min',
            min_delta=1e-12,
            cooldown=5,
            min_lr=1e-8
        )



        stop_lr_callback = StopWhenLRBelowThreshold(lr_threshold=5e-6, check_interval=5) # was -8



        history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=self.pdict['epochs'],
            batch_size=self.pdict['batchSize'],
            callbacks=[reduce_lr, stop_lr_callback]
        )


        self.model.save(self.datadir + self.model_name + 'trained_model_{}.keras'.format(tag))

        self.predvals = self.model.predict(self.X_test)
        self.testvals = self.y_test
        self.Xtestvals = self.X_test

        self.history_loss = history.history['loss']
        self.history_val_loss = history.history['val_loss']

        np.save(self.datadir + self.model_name + 'predicted_vals_{}.npy'.format(tag), self.predvals)
        np.save(self.datadir + self.model_name + 'test_vals_{}.npy'.format(tag), self.testvals)
        np.save(self.datadir + self.model_name + 'Xtest_vals_{}.npy'.format(tag), self.Xtestvals)

        np.save(self.datadir + self.model_name + 'history_loss_{}.npy'.format(tag), self.history_loss)
        np.save(self.datadir + self.model_name + 'history_val_loss_{}.npy'.format(tag), self.history_val_loss)

        if showplots:

            plt.figure(figsize=(4, 4))
            plt.clf()

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss - val=%.3g' % history.history['val_loss'][-1])
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')



            plt.savefig(self.datadir + self.model_name + 'savefigs/' +'loss_curve_lr:{}, bs:{}_{}.pdf'.format(pdict['learningRate'], pdict['batchSize'],tag))
            plt.close('all')




    def show_predictions(self, subtitle='', nshow=50, tag = ''):

        output_y = self.model.predict(self.X_train)
        self.predvals = self.model.predict(self.X_test)
        self.testvals = self.y_test
        self.Xtestvals = self.X_test

        MSE_val_totals = 0


        for i in range(np.shape(self.y_test)[0]):


            norm_im = self.y_test[i:i + 1, :, :, :]
            pred_im = self.predvals[i:i + 1, :, :, :]

            image_MSE = np.mean((norm_im - pred_im)**2)
            MSE_val_totals = MSE_val_totals + image_MSE

        MSE_val_totals = MSE_val_totals / np.shape(self.y_test)[0]


        ssim_val = 0

        for i in range(np.shape(self.y_test)[0]):
            current_ytest = self.y_test[i:i+1, :, :, :]
            current_predvals = self.predvals[i:i+1, :, :, :]
            ssim_val = ssim_val + pol_ssim(current_predvals, current_ytest)

        ssim_val = ssim_val/np.shape(self.y_test)[0]



        training_loss = np.load(self.datadir + self.model_name + 'history_loss_{}.npy'.format(tag))[-1]
        # MSE_val_totals = np.load(self.datadir + self.model_name + 'history_val_loss_{}.npy'.format(tag))[-1]



        tv_x = np.abs(np.diff(self.predvals, axis = 1))
        tv_y = np.abs(np.diff(self.predvals, axis = 2))
        total_tv = np.sum(tv_x) + np.sum(tv_y)

        temp_pred_it_stokes = self.predvals

        temp = temp_pred_it_stokes  + np.abs(np.min(temp_pred_it_stokes)) + 1e-15
        entropy = -np.sum(temp * np.log(temp))


        for k in range(nshow):
            plt.figure(figsize=(10, 9))
            plt.clf()

            current_predictedimage = self.predvals[k, :]  # PREDICTED IMAGES
            current_truevisibilities = self.Xtestvals[k, :]  # TRUTH VISIBILITIES
            current_truthimage = self.testvals[k, :]  # TRUTH IMAGES



            true_I_vis = (current_truevisibilities[0:153] * self.normfacts_X['sd_I_vis']) + self.normfacts_X[ 'mean_I_vis']
            true_I_cp = (current_truevisibilities[153:969] * self.normfacts_X['sd_I_cp']) +  self.normfacts_X['mean_I_cp']


            true_Q_vis = (current_truevisibilities[969:1122] * self.normfacts_X['sd_Q_vis']) + self.normfacts_X['mean_Q_vis']
            true_Q_cp = np.rad2deg((current_truevisibilities[1122:1938] * self.normfacts_X['sd_Q_cp']) +  self.normfacts_X['mean_Q_cp'])


            true_U_vis = (current_truevisibilities[1938:2091] * self.normfacts_X['sd_U_vis']) +  self.normfacts_X['mean_U_vis']
            true_U_cp = np.rad2deg((current_truevisibilities[2091:2907] * self.normfacts_X['sd_U_cp']) +  self.normfacts_X['mean_U_cp'])


            stokes_true = np.concatenate((current_truthimage, np.zeros((128, 128, 1))), axis = 2)
            stokes_true_unnorm = unnorm_images(stokes_true, self.normfacts_y)
            stokes_true_unnorm[:,:,0] = 10**(stokes_true_unnorm[:,:,0])

            stokes_pred = np.concatenate((current_predictedimage, np.zeros((128, 128, 1))), axis=2)
            stokes_pred_unnorm = unnorm_images(stokes_pred, self.normfacts_y)
            stokes_pred_unnorm[:, :, 0] = 10 ** (stokes_pred_unnorm[:, :, 0])

            pred_I_vis, pred_I_cp, pred_Q_vis, pred_Q_cp, pred_U_vis,  pred_U_cp = stokes_2_vampires_bispect(stokes_pred_unnorm, self.dftm_grid, self.indx_of_cp, self.bl, self.az)


            longest_baseline_in = np.max(self.indx_of_cp, axis = 0)
            longest_baselines = self.bl[longest_baseline_in ]

            plt.figure()
            plt.figure(figsize=(30, 8))
            plt.clf()

            plt.subplot(2, 6, 1)
            # max_lib = np.max([np.abs(np.min(stokes_true_unnorm[:,:,0])), np.abs(np.max(stokes_true_unnorm[:,:,0]))])
            tenth_maxlib = 0#(1/5)*max_lib
            plt.imshow(np.log10(stokes_true_unnorm[:,:,0]), cmap = 'seismic')# clim = [-max_lib + tenth_maxlib, max_lib - tenth_maxlib], cmap = 'seismic')
            plt.title('Stokes I - Truth')
            plt.colorbar(label = 'Normalised Flux')
            plt.xlabel('x (mas)')
            plt.ylabel('y (mas)')


            plt.subplot(2, 6, 7)

            # max_lib = np.max([np.abs(np.min(stokes_true_unnorm[:,:,0])), np.abs(np.max(stokes_true_unnorm[:,:,0]))])
            tenth_maxlib = 0#(1/5)*max_lib
            plt.imshow(np.log10(stokes_pred_unnorm[:,:,0]), cmap = 'seismic') # clim = [-max_lib + tenth_maxlib, max_lib - tenth_maxlib], cmap = 'seismic')
            plt.title('Stokes I - CNN Prediction')
            plt.colorbar(label = 'Normalised Flux')
            plt.xlabel('x (mas)')
            plt.ylabel('y (mas)')


            plt.subplot(2, 6, 2)
            plt.title("Stokes I vis - $V^2$ Correlation")
            plt.scatter(true_I_vis, pred_I_vis)
            line_x = np.arange(np.min(true_I_vis), np.max(true_I_vis), 0.001)
            line_y = line_x
            plt.plot(line_x, line_y, c='k')
            plt.xlabel('Stokes I - Predicted $V^2$')
            plt.ylabel('Stokes I - True $V^2$')
            plt.colorbar(label='Baseline Length (m)')

            plt.subplot(2, 6, 8)
            plt.title('Stokes I - CP Correlation')
            plt.scatter(true_I_cp, pred_I_cp)
            plt.xlabel('Stokes I - Predicted  (radians)')
            plt.ylabel('Stokes I - Real  (radians)')
            line_x = np.arange(np.min(true_I_cp), np.max(true_I_cp), (np.max(true_I_cp) -np.min(true_I_cp))/100 )
            line_y = line_x
            plt.plot(line_x, line_y, c='k')
            plt.colorbar(label='Baseline Length (m)')




            plt.subplot(2, 6, 3)
            max_lib = np.max([np.abs(np.min(stokes_true_unnorm[:,:,1])), np.abs(np.max(stokes_true_unnorm[:,:,1]))])
            tenth_maxlib = 0#(1/5)*max_lib
            plt.imshow(stokes_true_unnorm[:,:,1], clim = [-max_lib + tenth_maxlib, max_lib - tenth_maxlib], cmap = 'seismic')
            plt.title('Stokes Q - Truth')
            plt.colorbar(label = 'Normalised Flux')
            plt.xlabel('x (mas)')
            plt.ylabel('y (mas)')

            plt.subplot(2, 6, 9)
            plt.imshow(stokes_pred_unnorm[:,:,1], clim = [-max_lib + tenth_maxlib, max_lib - tenth_maxlib], cmap = 'seismic')
            plt.title('Stokes Q - CNN Prediction')
            plt.colorbar(label = 'Normalised Flux')
            plt.xlabel('x (mas)')
            plt.ylabel('y (mas)')

            plt.subplot(2, 6, 4)
            plt.title("Stokes Q - $V^2$ Correlation")
            plt.scatter(pred_Q_vis, true_Q_vis, c = self.bl, cmap = 'jet')
            line_x = np.arange(np.min(true_Q_vis), np.max(true_Q_vis), (np.max(true_Q_vis) -np.min(true_Q_vis))/100 )
            line_y = line_x
            plt.plot(line_x, line_y, c='k')
            plt.xlabel('Stokes Q - Predicted $V^2$')
            plt.ylabel('Stokes Q - True $V^2$')
            plt.colorbar(label='Baseline Length (m)')

            plt.subplot(2, 6, 10)
            plt.title('Stokes Q - CP Correlation')
            plt.scatter(pred_Q_cp, true_Q_cp, c = longest_baselines, cmap = 'jet')
            plt.xlabel('Stokes Q - Predicted CP (radians)')
            plt.ylabel('Stokes Q - Real CP (radians)')
            line_x = np.arange(np.min(true_Q_cp), np.max(true_Q_cp),  (np.max(true_Q_cp) -np.min(true_Q_cp))/100 )
            line_y = line_x
            plt.plot(line_x, line_y, c='k')
            plt.colorbar(label='Baseline Length (m)')

            plt.subplot(2, 6, 5)
            max_lib = np.max([np.abs(np.min(stokes_true_unnorm[:,:,2])), np.abs(np.max(stokes_true_unnorm[:,:,2]))])
            tenth_maxlib = 0#(1/5)*max_lib
            plt.imshow(stokes_true_unnorm[:,:,2], clim = [-max_lib + tenth_maxlib, max_lib - tenth_maxlib], cmap = 'seismic')
            plt.title('Stokes U - Truth')
            plt.colorbar(label = 'Normalised Flux')
            plt.xlabel('x (mas)')
            plt.ylabel('y (mas)')

            plt.subplot(2, 6, 11)
            plt.imshow(stokes_pred_unnorm[:,:,2], clim = [-max_lib + tenth_maxlib, max_lib - tenth_maxlib], cmap = 'seismic')
            plt.title('Stokes U - CNN Prediction')
            plt.colorbar(label = 'Normalised Flux')
            plt.xlabel('x (mas)')
            plt.ylabel('y (mas)')

            plt.subplot(2, 6, 6)
            plt.title("Stokes U - $V^2$ Correlation")
            plt.scatter(pred_U_vis, true_U_vis, c = self.bl, cmap = 'jet')
            line_x = np.arange(np.min(true_U_vis), np.max(true_U_vis),  (np.max(true_U_vis) -np.min(true_U_vis))/100 )
            line_y = line_x
            plt.plot(line_x, line_y, c='k')
            plt.xlabel('Stokes U - Predicted $V^2$')
            plt.ylabel('Stokes U - True $V^2$')
            plt.colorbar(label='Baseline Length (m)')

            plt.subplot(2, 6, 12)
            plt.title('Stokes U - CP Correlation')
            plt.scatter(pred_U_cp, true_U_cp, c = longest_baselines, cmap = 'jet')
            plt.xlabel('Stokes U - Predicted CP (radians)')
            plt.ylabel('Stokes U - Real CP (radians)')
            line_x = np.arange(np.min(true_U_cp), np.max(true_U_cp),  (np.max(true_U_cp) -np.min(true_U_cp))/100 )
            line_y = line_x
            plt.plot(line_x, line_y, c='k')
            plt.colorbar(label='Baseline Length (m)')

            plt.tight_layout()
            plt.savefig(self.datadir + self.model_name + 'savefigs/' +'output_full_{}_vis,{}.pdf'.format(k, subtitle))
            plt.close('all')


        return MSE_val_totals, ssim_val, total_tv, entropy, ssim_val


    def tune_model(self, lr, ep, pdict, true_obs= [], real_image=[], test_inputs_unnorm = [], true_err = [], MCFOST = False, tag = '', itmodel_path = '', example_num = 10, keeptrain = False, train_layers = []):

        def remove_dropout_from_model(pretrained_model):
            new_model = models.Sequential()

            # Iterate through the layers of the pre-trained model
            for layer in pretrained_model.layers:
                if not isinstance(layer, tf.keras.layers.Dropout):
                    # Clone the layer to avoid sharing weights
                    new_layer = models.clone_model(layer)
                    # Manually build the new layer to match the input shape
                    if not isinstance(layer, tf.keras.layers.InputLayer):
                        new_layer.build(layer.input_shape)
                        new_layer.set_weights(layer.get_weights())
                    new_model.add(new_layer)

            return new_model

        def freeze_dense_layers(pretrained_model):
            new_model = models.Sequential()

            # Iterate through the layers of the pre-trained model
            for layer in pretrained_model.layers:
                new_layer = models.clone_model(layer)  # Clone the layer to avoid sharing weights
                new_layer.build(layer.input_shape)  # Build the new layer to match input shape
                new_layer.set_weights(layer.get_weights())  # Copy the weights from the original layer

                # Freeze Dense layers by setting `trainable` to False
                if isinstance(layer, Dense):
                    new_layer.trainable = False  # False

                new_model.add(new_layer)

            return new_model

        def freeze_transpose_conv(pretrained_model, num_lay_train):
            new_model = models.clone_model(pretrained_model)  # Clone architecture
            new_model.set_weights(pretrained_model.get_weights())  # Copy weights

            # Find all Conv2DTranspose layers
            transpose_conv_layers = [layer for layer in new_model.layers if isinstance(layer, layers.Conv2DTranspose)]

            # Get the number of Conv2DTranspose layers
            total_conv_layers = len(transpose_conv_layers)

            # Freeze all but the last two Conv2DTranspose layers
            for i, layer in enumerate(transpose_conv_layers):
                print(i)
                if i >= total_conv_layers - num_lay_train:   # freezes first layers, leaves last 6 trainable
                    layer.trainable = False


            return new_model


        from tensorflow.keras.models import clone_model
        from tensorflow.keras import regularizers
        from tensorflow.keras.models import clone_model

        class LearningRateLogger(tf.keras.callbacks.Callback):
            def __init__(self, threshold, save_path):
                super(LearningRateLogger, self).__init__()
                self.threshold = threshold  # Learning rate threshold
                self.save_path = save_path  # Path to save the model
                self.epoch_counter = 0  # Counter for epochs

            def on_epoch_end(self, epoch, logs=None):
                self.epoch_counter += 1  # Increment epoch counter

                # Only monitor every 10 epochs
                if self.epoch_counter % 25 == 0:
                    lr = self.model.optimizer.lr
                    if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                        lr = lr(epoch)
                    else:
                        lr = lr.numpy()
                    self.lrs.append(lr)

                   # Stop training and save model if learning rate is below the threshold
                    if lr < self.threshold:
                        print(f"Stopping training as learning rate fell below {self.threshold}.")
                        self.model.stop_training = True
                        self.model.save(self.save_path)
                        print(f"Model saved to {self.save_path}")

            def on_train_begin(self, logs=None):
                self.lrs = []  # Initialize learning rate list
                self.epoch_counter = 0


        class CustomLRScheduler(tf.keras.callbacks.Callback):
            def __init__(self, patience=5, factor=0.5, min_lr=1e-6):
                super().__init__()
                self.patience = patience  # Number of epochs to wait for improvement
                self.factor = factor  # Factor by which to reduce the learning rate
                self.min_lr = min_lr  # Minimum learning rate
                self.wait = 0  # Epochs without improvement
                self.best_loss = float('inf')  # Track the best validation loss

            def on_epoch_end(self, epoch, logs=None):
                current_loss = logs.get('val_loss')  # Monitor the validation loss

                # If the current loss is not better than the best one, increment the wait counter
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.wait = 0
                else:
                    self.wait += 1

                # If the patience is reached, decrease the learning rate
                if self.wait >= self.patience:
                    current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                    new_lr = max(current_lr * self.factor, self.min_lr)
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                    self.wait = 0  # Reset wait counter
                    print(f"\nEpoch {epoch + 1}: Learning rate reduced to {new_lr:.6f}")


        def compute_ewc_importance(model, dataset, loss_fn, num_batches=None):
            """
            Computes Fisher Information and stores old parameters for EWC.

            Args:
                model: Trained Keras model (e.g., your self.model)
                dataset: tf.data.Dataset yielding (x, y)
                loss_fn: loss function used on Task A (e.g., tf.keras.losses.CategoricalCrossentropy())
                num_batches: Optional limit on how many batches to use

            Returns:
                old_params_dict: dict of parameter values (tf.Tensor)
                fisher_dict: dict of Fisher information (tf.Tensor)
            """


            old_params_dict = {var.ref(): tf.identity(var) for var in model.trainable_variables}
            fisher_dict = {var.ref(): tf.zeros_like(var) for var in model.trainable_variables}

            sample_count = 0

            for batch_idx, (x_batch, y_batch) in enumerate(dataset):
                if num_batches and batch_idx >= num_batches:
                    break

                with tf.GradientTape() as tape:
                    preds = model(x_batch, training=False)

                    y_batch = tf.cast(y_batch, tf.float32)
                    preds = tf.cast(preds, tf.float32)

                    loss = loss_fn(y_batch, preds)
                    loss = tf.reduce_mean(loss)

                grads = tape.gradient(loss, model.trainable_variables)

                for var, grad in zip(model.trainable_variables, grads):
                    if grad is not None and fisher_dict[var.ref()].shape == grad.shape:
                        fisher_dict[var.ref()] += tf.square(grad)

                sample_count += 1

            # Normalize Fisher info
            for ref in fisher_dict:
                fisher_dict[ref] /= float(sample_count)

            return old_params_dict, fisher_dict


        true_obs_norm = true_obs
        true_images_norm = real_image
        true_obs_unnorm = test_inputs_unnorm

        trained_model = self.model
        cnn_pred_image_norm = trained_model.predict(true_obs_norm)
        new_model = remove_dropout_from_model(trained_model)


        loss_fn = custom_image_with_arguments()
        batch_size = 32  # or whatever batch size you want
        taskA_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        taskA_dataset = taskA_dataset.batch(batch_size)
        taskA_dataset = taskA_dataset.prefetch(tf.data.AUTOTUNE)
        taskA_dataset = taskA_dataset.shuffle(buffer_size=1000).take(100)


        old_params_dict, fisher_dict = compute_ewc_importance(
            model=new_model,
            dataset=taskA_dataset,
            loss_fn=loss_fn,
            num_batches=100  # optional (use fewer batches if dataset is large)
        )

        # print this out and check what the size of old_params_dict, fisher_dict

        # new_model = add_kernel_regularization(new_model, reg_type="l1", value=self.reg_strength)
        # new_model = freeze_dense_layers(new_model)
        # new_model = freeze_transpose_conv(new_model, self.num_conv_freeze)


        lr_logger = LearningRateLogger(threshold=5e-15, save_path= self.datadir + self.model_name + 'it_fit_model_{}.keras'.format(tag))
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=5, verbose=0, mode='min', min_delta=1e-7, cooldown=5, min_lr=1e-15)

        # fisher_dict = 0
        # old_params_dict = 0


        new_model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                          loss=custom_loss_with_arguments(self.dftm_grid, self.indx_of_cp,
                          self.normfacts_y, true_obs_norm, true_err, self.normfacts_X, self.reg_strength,
                                                          fisher_dict, old_params_dict, new_model)) # add some errors here




        history = new_model.fit(true_obs_norm, cnn_pred_image_norm, validation_data=(true_obs_norm, cnn_pred_image_norm), epochs=ep, batch_size=1, callbacks = [lr_logger, reduce_lr])
        new_model.save(self.datadir + self.model_name + 'it_fit_model_{}.keras'.format(tag))

        ewc_loss = 0.0
        for var in new_model.trainable_variables:
            ref = var.ref()
            if ref in fisher_dict:
                fisher = fisher_dict[ref]
                old_param = old_params_dict[ref]
                param_diff = var - old_param
                ewc_loss += tf.reduce_sum(fisher * tf.square(param_diff))


        np.save(self.datadir + self.model_name + 'savefigs/' + '{}_ewc_loss'.format(tag), ewc_loss)


        plt.figure(figsize = (6,4))
        plt.subplot(1,2,1)
        plt.plot(np.log10(lr_logger.lrs))
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate over Epochs')

        plt.subplot(1,2,2)
        plt.plot(np.log10(history.history['loss']))
        plt.title('Model loss - =%.3g' % history.history['loss'][-1])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')

        plt.savefig(  self.datadir + self.model_name + 'savefigs/' + '{}_ITERATIVEloss_curve_ .pdf'.format(tag ))
        plt.close('all')



        it_pred_image_norm = new_model.predict(true_obs_norm)  # was X_once_i
        I = np.expand_dims(true_images_norm[0,0,], axis = 2)
        Q = np.expand_dims(true_images_norm[0,1,], axis = 2)
        U = np.expand_dims(true_images_norm[0,2,], axis = 2)

        true_stokes = np.concatenate((I,Q,U), axis = 2)
        true_stokes = np.expand_dims(true_stokes, axis = 0)

        real_stokes = np.concatenate((true_stokes[0:1,], np.zeros((1, 128, 128, 1))), axis=3)
        #true_images_unnorm = unnor#m_images(real_stokes, self.normfacts_y)
        true_images_unnorm = real_stokes #[0, :, :, 0] = 10 ** (true_images_unnorm[0, :, :, 0])

        pred_trained_stokes = np.concatenate((cnn_pred_image_norm[0:1,], np.zeros((1, 128, 128, 1))), axis=3)
        cnn_pred_image_unnorm = unnorm_images(pred_trained_stokes, self.normfacts_y)
        cnn_pred_image_unnorm[0, :, :, 0] = 10 ** (cnn_pred_image_unnorm[0, :, :, 0])

        pred_it_stokes = np.concatenate((it_pred_image_norm[0:1,], np.zeros((1, 128, 128, 1))), axis=3)
        it_pred_image_unnorm = unnorm_images(pred_it_stokes, self.normfacts_y)
        it_pred_image_unnorm[0, :, :, 0] = 10 ** (it_pred_image_unnorm[0, :, :, 0])




        plt.figure()
        plt.subplot(3,3,1)
        plt.imshow(np.log10(true_images_unnorm[0, :, :, 0]), cmap = 'seismic')
        plt.colorbar()
        plt.subplot(3,3,2)
        tep = np.max(np.array([np.abs(np.min(true_images_unnorm[0, :, :, 1])), np.abs(np.max(true_images_unnorm[0, :, :, 1]))]))
        plt.imshow( true_images_unnorm[0, :, :, 1], cmap = 'seismic', clim = [-tep, tep])
        plt.colorbar()

        plt.subplot(3,3,3)
        tep = np.max(np.array([np.abs(np.min(true_images_unnorm[0, :, :, 2])), np.abs(np.max(true_images_unnorm[0, :, :, 2]))]))
        plt.imshow( true_images_unnorm[0, :, :, 2], cmap = 'seismic', clim = [-tep, tep])
        plt.colorbar()

        plt.subplot(3, 3, 4)
        plt.imshow(np.log10(cnn_pred_image_unnorm[0, :, :, 0]), cmap = 'seismic')
        plt.colorbar()
        plt.subplot(3, 3, 5)
        tep = np.max(np.array([np.abs(np.min(cnn_pred_image_unnorm[0, :, :, 1])), np.abs(np.max(cnn_pred_image_unnorm[0, :, :, 1]))]))
        plt.imshow(cnn_pred_image_unnorm[0, :, :, 1], cmap = 'seismic', clim = [-tep, tep])
        plt.colorbar()
        plt.subplot(3, 3, 6)
        tep = np.max(np.array([np.abs(np.min(cnn_pred_image_unnorm[0, :, :, 2])), np.abs(np.max(cnn_pred_image_unnorm[0, :, :, 2]))]))
        plt.imshow(cnn_pred_image_unnorm[0, :, :, 2], cmap = 'seismic', clim = [-tep, tep])
        plt.colorbar()
        plt.subplot(3, 3, 7)
        plt.imshow(np.log10(it_pred_image_unnorm[0, :, :, 0]), cmap = 'seismic')
        plt.colorbar()
        plt.subplot(3, 3, 8)
        tep = np.max(np.array([np.abs(np.min(it_pred_image_unnorm[0, :, :, 1])), np.abs(np.max(it_pred_image_unnorm[0, :, :, 1]))]))
        plt.imshow(it_pred_image_unnorm[0, :, :, 1], cmap = 'seismic', clim = [-tep, tep])
        plt.colorbar()

        plt.subplot(3, 3, 9)
        tep = np.max(np.array([np.abs(np.min(it_pred_image_unnorm[0, :, :, 2])), np.abs(np.max(it_pred_image_unnorm[0, :, :, 2]))]))
        plt.imshow(it_pred_image_unnorm[0, :, :, 2], cmap = 'seismic', clim = [-tep, tep])
        plt.colorbar()


        plt.savefig(  self.datadir + self.model_name + 'savefigs/' + '{}_images_ .pdf'.format(tag ))
        plt.close('all')

        np.save(self.datadir + self.model_name + '{}_true_images_unnorm'.format(tag), true_images_unnorm)
        np.save(self.datadir + self.model_name + '{}_cnn_pred_images_unnorm'.format(tag), cnn_pred_image_unnorm)
        np.save(self.datadir + self.model_name + '{}_it_pred_image_unnorm'.format(tag), it_pred_image_unnorm)


        real_I_vis = true_obs_unnorm[0, 0:153]
        real_I_cp = true_obs_unnorm[0,153:969]
        real_Q_vis = true_obs_unnorm[0,969:1122]
        real_Q_cp = true_obs_unnorm[0,1122:1938]
        real_U_vis = true_obs_unnorm[0,1938:2091]
        real_U_cp = true_obs_unnorm[0,2091:2907]


        pred_I_vis, pred_I_cp, pred_Q_vis, pred_Q_cp, pred_U_vis,  pred_U_cp = stokes_2_vampires_bispect(cnn_pred_image_unnorm[0,], self.dftm_grid, self.indx_of_cp, self.bl, self.az)
        it_I_vis,     it_I_cp,   it_Q_vis, it_Q_cp,    it_U_vis, it_U_cp  = stokes_2_vampires_bispect(it_pred_image_unnorm[0,], self.dftm_grid, self.indx_of_cp, self.bl, self.az)

       # true_obs_norm
        it_I_vis_N, it_I_cp_N,  it_Q_vis_N, it_Q_cp_N, it_U_vis_N, it_U_cp_N = norm_observables_tensor(it_I_vis, it_I_cp, it_Q_vis, it_Q_cp, it_U_vis, it_U_cp, self.normfacts_X)

        it_obs_norm = np.concatenate((it_I_vis_N, it_I_cp_N,  it_Q_vis_N, it_Q_cp_N, it_U_vis_N, it_U_cp_N), axis = 0)

        MSE_obs = np.sum((true_obs_norm - it_obs_norm)**2)
        np.save(self.datadir + self.model_name + 'savefigs/' + '{}_MSE_obs'.format(tag), MSE_obs)




        plt.figure(figsize=(23, 18))

        plt.subplot(6, 6, 1)
        # plt.errorbar(self.bl, real_I_vis, yerr=self.pdict['true_I_vis_err'] , fmt='none', ecolor='grey', elinewidth=2)
        plt.scatter(self.bl, real_I_vis)
        plt.title('True I Vis')
        plt.xlabel('Baseline Length (m)')
        plt.ylabel('Visibility')

        plt.subplot(6, 6, 2)
        # plt.errorbar(self.az, real_Q_vis, yerr=self.pdict['true_Q_vis_err'] , fmt='none', ecolor='grey', elinewidth=2)
        plt.scatter(self.az, real_Q_vis, c=self.bl, cmap = 'jet')

        plt.title('True Q Vis')
        plt.xlabel('Azimuth Angle (rad)')
        plt.ylabel('Visibility')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 3)

        # plt.errorbar(self.az, real_U_vis, yerr=self.pdict['true_U_vis_err'] , fmt='none', ecolor='grey', elinewidth=2)
        plt.scatter(self.az, real_U_vis, c=self.bl, cmap = 'jet')
        plt.title('True U Vis')
        plt.xlabel('Azimuth Angle (rad)')
        plt.ylabel('Visibility')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 4)
        plt.scatter(real_I_vis, real_I_vis, c = self.bl, cmap = 'jet')
        line_x = np.arange(np.min(real_I_vis), np.max(real_I_vis), (np.max(real_I_vis) - np.min(real_I_vis)) / 100)
        line_y = line_x
        plt.plot(line_x, line_y, c='k')
        plt.xlabel('True')
        plt.ylabel('True')
        plt.title('True vs True I vis')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 5)
        plt.scatter(real_Q_vis, real_Q_vis, c = self.bl, cmap = 'jet')
        line_x = np.arange(np.min(real_Q_vis), np.max(real_Q_vis), (np.max(real_Q_vis) - np.min(real_Q_vis)) / 100)
        line_y = line_x
        plt.plot(line_x, line_y, c='k')
        plt.xlabel('True')
        plt.ylabel('True')
        plt.title('True vs True Q vis')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 6)
        plt.scatter(real_U_vis, real_U_vis, c = self.bl, cmap = 'jet')
        line_x = np.arange(np.min(real_U_vis), np.max(real_U_vis), (np.max(real_U_vis) - np.min(real_U_vis)) / 100)
        line_y = line_x
        plt.plot(line_x, line_y, c='k')
        plt.xlabel('True')
        plt.ylabel('True')
        plt.title('True vs True U vis')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 7)
        plt.scatter(self.bl, pred_I_vis)
        plt.title('Trained I vis')
        plt.ylabel('Visibility')
        plt.xlabel('Baseline Length (m)')


        plt.subplot(6, 6, 8)
        plt.scatter(self.az, pred_Q_vis, c=self.bl, cmap = 'jet')
        plt.title('Trained Q vis')
        plt.xlabel('Azimuth Angle (rad)')
        plt.ylabel('Visibility')
        plt.colorbar(label = 'Baseline (m)')


        plt.subplot(6, 6, 9)
        plt.scatter(self.az, pred_U_vis, c=self.bl, cmap = 'jet')
        plt.title('Trained U vis')
        plt.xlabel('Azimuth Angle (rad)')
        plt.ylabel('Visibility')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 10)
        plt.scatter(real_I_vis, pred_I_vis, c = self.bl, cmap = 'jet')
        plt.title('Stokes I - True vs CNN Predicted $V^2$')
        line_x = np.arange(np.min(real_I_vis), np.max(real_I_vis), (np.max(real_I_vis) - np.min(real_I_vis)) / 100)
        line_y = line_x
        plt.plot(line_x, line_y, c='k')
        plt.xlabel('True $V^2$')
        plt.ylabel('CNN Predicted $V^2$')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 11)
        plt.scatter(real_Q_vis, pred_Q_vis, c = self.bl, cmap = 'jet')
        plt.title('Stokes Q - True vs CNN Predicted $V^2$')
        plt.xlabel('True $V^2$')
        line_x = np.arange(np.min(real_Q_vis), np.max(real_Q_vis), (np.max(real_Q_vis) - np.min(real_Q_vis)) / 100)
        line_y = line_x
        plt.plot(line_x, line_y, c='k')
        plt.ylabel('CNN Predicted $V^2$')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 12)
        plt.scatter(real_U_vis, pred_U_vis, c = self.bl, cmap = 'jet')
        plt.title('Stokes U - True vs CNN Predicted $V^2$')
        line_x = np.arange(np.min(real_U_vis), np.max(real_U_vis), (np.max(real_U_vis) - np.min(real_U_vis)) / 100)
        line_y = line_x
        plt.plot(line_x, line_y, c='k')
        plt.xlabel('True $V^2$')
        plt.ylabel('CNN Predicted $V^2$')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 13)
        plt.scatter(self.bl, it_I_vis)
        plt.title('Iterative I vis')
        plt.xlabel('Baseline (m)')
        plt.ylabel('Visibility')

        plt.subplot(6, 6, 14)
        plt.scatter(self.az, it_Q_vis, c=self.bl, cmap = 'jet')
        plt.title('Iterative Q vis')
        plt.xlabel('Azimuth Angle (rad)')
        plt.ylabel('Visibility')
        plt.colorbar(label = 'Baseline (m)')


        plt.subplot(6, 6, 15)
        plt.scatter(self.az, it_U_vis, c=self.bl, cmap = 'jet')
        plt.title('Iterative U vis')
        plt.xlabel('Azimuth Angle (rad)')
        plt.ylabel('Visibility')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 16)
        plt.scatter(real_I_vis, it_I_vis, c = self.bl, cmap = 'jet')
        plt.title('Stokes I - True vs Iterative Fitted $V^2$')
        line_x = np.arange(np.min(real_I_vis), np.max(real_I_vis), (np.max(real_I_vis) - np.min(real_I_vis)) / 100)
        line_y = line_x
        plt.plot(line_x, line_y, c='k')
        plt.xlabel('True $V^2$')
        plt.ylabel('Iterative Fitted $V^2$')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 17)
        plt.scatter(real_Q_vis, it_Q_vis, c = self.bl, cmap = 'jet')
        line_x = np.arange(np.min(real_Q_vis), np.max(real_Q_vis), (np.max(real_Q_vis) - np.min(real_Q_vis)) / 100)
        line_y = line_x
        plt.plot(line_x, line_y, c='k')
        plt.title('Stokes Q - True vs Iterative Fitted $V^2$')
        plt.xlabel('True $V^2$')
        plt.ylabel('Iterative Fitted $V^2$')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 18)
        plt.scatter(real_U_vis, it_U_vis, c = self.bl, cmap = 'jet')
        line_x = np.arange(np.min(real_U_vis), np.max(real_U_vis), (np.max(real_U_vis) - np.min(real_U_vis)) / 100)
        line_y = line_x
        plt.plot(line_x, line_y, c='k')
        plt.title('Stokes U - True vs Iterative Fitted $V^2$')
        plt.xlabel('True $V^2$')
        plt.ylabel('Iterative Fitted $V^2$')
        plt.colorbar(label = 'Baseline (m)')

        ######################################################################################################################
        #######################################################################################################################################

        maxbl = np.max(self.indx_of_cp, axis=0)
        max_bl = self.bl[maxbl]

        plt.subplot(6, 6, 19)
        plt.scatter(max_bl, real_I_cp)
        plt.title('True I imag')
        plt.xlabel("Max Baseline (m)")

        plt.subplot(6, 6, 20)
        plt.hist(real_Q_cp, bins=100)
        plt.title('True Q CP')
        plt.xlabel('Q CP')
        plt.ylabel('Freq')

        plt.subplot(6, 6, 21)
        plt.hist(real_U_cp, bins=100)
        plt.title('True U')
        plt.xlabel('U CP')
        plt.ylabel('Freq')

        plt.subplot(6, 6, 22)
        plt.scatter(real_I_cp, real_I_cp, c = max_bl, cmap = 'jet')
        plt.title('True vs True I imag')
        # line_x = np.arange(np.min(real_I_cp), np.max(real_I_cp), (np.max(real_I_cp) - np.min(real_I_cp)) / 100)
        # line_y = line_x
        # plt.plot(line_x, line_y, c='k')
        plt.xlabel('True')
        plt.ylabel('True')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 23)
        plt.scatter(real_Q_cp, real_Q_cp, c = max_bl, cmap = 'jet')
        plt.title('True vs True Q cp')
        line_x = np.arange(np.min(real_Q_cp), np.max(real_Q_cp), (np.max(real_Q_cp) - np.min(real_Q_cp)) / 100)
        line_y = line_x
        plt.plot(line_x, line_y, c='k')
        plt.xlabel('True')
        plt.ylabel('True')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 24)
        plt.scatter(real_U_cp, real_U_cp, c = max_bl, cmap = 'jet')
        plt.title('True vs True U cp')
        line_x = np.arange(np.min(real_U_cp), np.max(real_U_cp), (np.max(real_U_cp) - np.min(real_U_cp)) / 100)
        line_y = line_x
        plt.plot(line_x, line_y, c='k')
        plt.xlabel('True')
        plt.ylabel('True')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 25)
        plt.scatter(max_bl, pred_I_cp, c = max_bl, cmap = 'jet')
        plt.xlabel("Max Baseline (m)")
        plt.title('Trained I')
        plt.ylabel('I imag bispec')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 26)
        plt.hist(pred_Q_cp, bins=100)
        plt.title('Trained Q')
        plt.xlabel('Q CP')
        plt.ylabel('Freq')

        plt.subplot(6, 6, 27)
        plt.hist(pred_U_cp, bins=100)
        plt.title('Trained U')
        plt.xlabel('U CP')
        plt.ylabel('Freq')

        plt.subplot(6, 6, 28)
        plt.scatter(real_I_cp, pred_I_cp, c = max_bl, cmap = 'jet')
        plt.title('Stokes I - True vs CNN Predicted I CP')

        # line_x = np.arange(np.min(real_I_cp), np.max(real_I_cp), (np.max(real_I_cp) - np.min(real_I_cp)) / 100)
        # line_y = line_x
        # plt.plot(line_x, line_y, c='k')
        plt.xlabel('True I CP')
        plt.colorbar(label = 'Baseline (m)')
        plt.ylabel('CNN Predicted I CP')

        plt.subplot(6, 6, 29)
        plt.scatter(real_Q_cp, pred_Q_cp, c = max_bl, cmap = 'jet')
        plt.title('Stokes Q - True vs CNN Predicted CP')
        plt.xlabel('True CP')
        line_x = np.arange(np.min(real_Q_cp), np.max(real_Q_cp), (np.max(real_Q_cp) - np.min(real_Q_cp)) / 100)
        line_y = line_x
        plt.plot(line_x, line_y, c='k')
        plt.ylabel('CNN Predicted CP')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 30)
        plt.scatter(real_U_cp, pred_U_cp, c = max_bl, cmap = 'jet')
        plt.title('Stokes U - True vs CNN Predicted CP')
        line_x = np.arange(np.min(real_U_cp), np.max(real_U_cp), (np.max(real_U_cp) - np.min(real_U_cp)) / 100)
        line_y = line_x
        plt.plot(line_x, line_y, c='k')
        plt.xlabel('True CP')
        plt.ylabel('CNN Predicted CP')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 31)
        plt.scatter(max_bl, it_I_cp)
        plt.xlabel("Max Baseline (m)")
        plt.title('Iterative I')
        plt.ylabel('I imag bisp.')
        plt.xlabel('Max bl (m)')

        plt.subplot(6, 6, 32)
        plt.hist(it_Q_cp, bins=100)
        plt.title('Iterative Q')
        plt.xlabel('Q CP')
        plt.ylabel('Freq')

        plt.subplot(6, 6, 33)
        plt.hist(it_U_cp, bins=100)
        plt.title('Iterative U')
        plt.xlabel('U CP')
        plt.ylabel('Freq')

        plt.subplot(6, 6, 34)
        plt.scatter(real_I_cp, it_I_cp, c = max_bl, cmap = 'jet')
        plt.title('Stokes I - True vs Iterative Fit I CP')
        # line_x = np.arange(np.min(real_I_cp), np.max(real_I_cp), (np.max(real_I_cp) - np.min(real_I_cp)) / 100)
        # line_y = line_x
        # plt.plot(line_x, line_y, c='k')
        plt.xlabel('True I CP')
        plt.ylabel('Iterative Fit CP')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 35)
        plt.scatter(real_Q_cp, it_Q_cp, c = max_bl, cmap = 'jet')
        line_x = np.arange(np.min(real_Q_cp), np.max(real_Q_cp), (np.max(real_Q_cp) - np.min(real_Q_cp)) / 100)
        line_y = line_x
        plt.plot(line_x, line_y, c='k')
        plt.title('Stokes Q - True vs Iterative Fit CP')
        plt.xlabel('True CP')
        plt.ylabel('Iterative Fit CP')
        plt.colorbar(label = 'Baseline (m)')

        plt.subplot(6, 6, 36)
        plt.scatter(real_U_cp, it_U_cp, c = max_bl, cmap = 'jet')
        line_x = np.arange(np.min(real_U_cp), np.max(real_U_cp), (np.max(real_U_cp) - np.min(real_U_cp)) / 100)
        line_y = line_x
        plt.plot(line_x, line_y, c='k')
        plt.title('Stokes U - True vs Iterative Fit CP')
        plt.xlabel('True CP')
        plt.ylabel('Iterative Fit CP')
        plt.colorbar(label = 'Baseline (m)')

        plt.tight_layout()
        plt.savefig(self.datadir + self.model_name + 'savefigs/' +'{}_Real_Pred_Iterative_vis.pdf'.format(tag))
        plt.close('all')

        true_rem = true_images_unnorm[0,:,:,0].sum()
        true_images_unnorm[0, :, :, 0] = true_images_unnorm[0,:,:,0]/true_rem
        true_images_unnorm[0, :, :, 1] = true_images_unnorm[0,:,:,1]/true_rem
        true_images_unnorm[0, :, :, 2] = true_images_unnorm[0,:,:,2]/true_rem

        pred_rem = cnn_pred_image_unnorm[0, :, :, 0].sum()
        cnn_pred_image_unnorm[0, :, :, 0] = cnn_pred_image_unnorm[0, :, :, 0] / pred_rem
        cnn_pred_image_unnorm[0, :, :, 1] = cnn_pred_image_unnorm[0, :, :, 1] / pred_rem
        cnn_pred_image_unnorm[0, :, :, 2] = cnn_pred_image_unnorm[0, :, :, 2] / pred_rem

        it_rem = it_pred_image_unnorm[0, :, :, 0].sum()
        it_pred_image_unnorm[0, :, :, 0] = it_pred_image_unnorm[0, :, :, 0] / it_rem
        it_pred_image_unnorm[0, :, :, 1] = it_pred_image_unnorm[0, :, :, 1] / it_rem
        it_pred_image_unnorm[0, :, :, 2] = it_pred_image_unnorm[0, :, :, 2] / it_rem

        it_MSE = np.mean((it_pred_image_unnorm - true_images_unnorm) ** 2)

        pol_SS = pol_ssim(it_pred_image_unnorm, true_images_unnorm)


        return it_MSE, pol_SS



    def iterative_fitting_example(self, image):
        
        self.pdict['images_path'] = image

        input_stokes = np.load(self.datadir + '{}'.format(self.pdict['images_path']))
        print(np.shape(input_stokes))

        new_imI = np.expand_dims(gaussian_filter(input_stokes[0,], sigma=1.5),  axis=0)
        new_imQ = np.expand_dims(gaussian_filter(input_stokes[1,], sigma=1.5),axis=0)
        new_imU = np.expand_dims(gaussian_filter(input_stokes[2,], sigma=1.5),axis=0)
        new_imV = np.zeros(np.shape(new_imI))

        new_stokes = np.concatenate((new_imI, new_imQ, new_imU, new_imV), axis=0)
        new_stokes = np.pad(new_stokes, pad_width=((0, 0), (13, 14), (13, 14)), mode='constant', constant_values=0)
        new_stokes, w = window_amical(new_stokes, 40, m=5)

        test_loc = np.argwhere(new_stokes[0, :, :] < 1e-15)
        new_stokes[0, test_loc[:, 0], test_loc[:, 1]] = 1e-15
        sum_stokes_I = np.sum(new_stokes[0,])

        new_stokes[0,] = new_stokes[0,] / sum_stokes_I
        new_stokes[1,] = new_stokes[1,] / sum_stokes_I
        new_stokes[2,] = new_stokes[2,] / sum_stokes_I
        new_stokes[3,] = 0

        input_stokes = new_stokes
        real_image = input_stokes
        obs, err = stokes_2_vampires_bispect_noise(real_image, self.dftm_grid, self.indx_of_cp, self.bl, self.az)

        true_I_vis = obs[0:153]
        true_I_cp = obs[153:969]
        true_Q_vis = obs[969:1122]
        true_Q_cp = obs[1122:1938]
        true_U_vis = obs[1938:2091]
        true_U_cp = obs[2091:2907]

        true_I_vis_err = err[0:153]
        true_I_cp_err = err[153:969]
        true_Q_vis_err = err[969:1122]
        true_Q_cp_err = err[1122:1938]
        true_U_vis_err = err[1938:2091]
        true_U_cp_err = err[2091:2907]

        obs = np.concatenate((true_I_vis, true_I_cp, true_Q_vis, true_Q_cp, true_U_vis, true_U_cp), axis=0)

        test_inputs_unnorm = np.expand_dims(obs, axis=0)
        test_inputs_unnorm = tf.convert_to_tensor(test_inputs_unnorm, dtype=tf.float32)

        true_I_vis = (true_I_vis - self.normfacts_X['mean_I_vis']) / self.normfacts_X['sd_I_vis']
        true_I_cp = (true_I_cp - self.normfacts_X['mean_I_cp']) / self.normfacts_X['sd_I_cp']

        true_Q_vis = (true_Q_vis - self.normfacts_X['mean_Q_vis']) / self.normfacts_X['sd_Q_vis']
        true_Q_cp = (true_Q_cp - self.normfacts_X['mean_Q_cp']) / self.normfacts_X['sd_Q_cp']

        true_U_vis = (true_U_vis - self.normfacts_X['mean_U_vis']) / self.normfacts_X['sd_U_vis']
        true_U_cp = (true_U_cp - self.normfacts_X['mean_U_cp']) / self.normfacts_X['sd_U_cp']

        true_I_vis_err = true_I_vis_err / self.normfacts_X['sd_I_vis']
        true_Q_vis_err = true_Q_vis_err / self.normfacts_X['sd_Q_vis']
        true_U_vis_err = true_U_vis_err / self.normfacts_X['sd_U_vis']

        true_I_cp_err = true_I_cp_err / self.normfacts_X['sd_I_cp']
        true_Q_cp_err = true_Q_cp_err / self.normfacts_X['sd_Q_cp']
        true_U_cp_err = true_U_cp_err / self.normfacts_X['sd_U_cp']

        plt.figure(figsize=(10, 8))

        plt.subplot(2, 6, 1)
        plt.scatter(self.bl, true_I_vis)
        plt.subplot(2, 6, 2)
        plt.scatter(self.az, true_Q_vis, c=self.bl)
        plt.subplot(2, 6, 3)
        plt.scatter(self.az, true_U_vis, c=self.bl)
        plt.subplot(2, 6, 4)
        lip = np.arange(0, len(true_I_cp))
        plt.scatter(lip, true_I_cp, s=1)
        plt.subplot(2, 6, 5)
        plt.scatter(lip, true_Q_cp, s=1)
        plt.subplot(2, 6, 6)
        plt.scatter(lip, true_U_cp, s=1)

        plt.subplot(2, 6, 7)
        plt.scatter(self.bl, true_I_vis, s=1)
        plt.errorbar(self.bl, true_I_vis, yerr=true_I_vis_err, fmt='', linestyle='none', color='black')
        plt.subplot(2, 6, 8)
        plt.scatter(self.az, true_Q_vis, c=self.bl, s=1)
        plt.errorbar(self.az, true_Q_vis, yerr=true_Q_vis_err, fmt='', linestyle='none', color='black')
        plt.subplot(2, 6, 9)
        plt.scatter(self.az, true_U_vis, c=self.bl, s=1)
        plt.errorbar(self.az, true_U_vis, yerr=true_U_vis_err, fmt='', linestyle='none', color='black')
        plt.subplot(2, 6, 10)
        lip = np.arange(0, len(true_I_cp))
        plt.scatter(lip, true_I_cp, s=1)
        plt.errorbar(lip, true_I_cp, yerr=true_I_cp_err, fmt='', linestyle='none', color='black')
        plt.subplot(2, 6, 11)
        plt.scatter(lip, true_Q_cp, s=1)
        plt.errorbar(lip, true_Q_cp, yerr=true_Q_cp_err, fmt='', linestyle='none', color='black')
        plt.subplot(2, 6, 12)
        plt.scatter(lip, true_U_cp, s=1)
        plt.errorbar(lip, true_U_cp, yerr=true_U_cp_err, fmt='', linestyle='none', color='black')

        plt.savefig(self.datadir + self.model_name + 'savefigs/' + 'dump_{}.pdf'.format(self.pdict['images_path']))
        plt.close('all')

        true_obs = np.expand_dims(np.concatenate((true_I_vis, true_I_cp, true_Q_vis, true_Q_cp, true_U_vis, true_U_cp)),  axis=0)
        true_err = np.expand_dims( np.concatenate((true_I_vis_err, true_I_cp_err, true_Q_vis_err, true_Q_cp_err, true_U_vis_err, true_U_cp_err)),
            axis=0)

        np.save(self.datadir + self.model_name + 'savefigs/' + 'true_obs_{}.pdf'.format(self.pdict['images_path']),
                true_obs)
        np.save(self.datadir + self.model_name + 'savefigs/' + 'true_err_{}.pdf'.format(self.pdict['images_path']),
                true_err)

        test_inputs_unnorm = tf.convert_to_tensor(test_inputs_unnorm, dtype=tf.float32)
        true_obs = tf.convert_to_tensor(true_obs, dtype=tf.float32)

        true_obs = np.tile(true_obs, (self.pdict['num_reps'], 1))
        test_inputs_unnorm = np.tile(test_inputs_unnorm, (self.pdict['num_reps'], 1))
        real_image = np.expand_dims(real_image, axis=0)
        real_image = np.tile(real_image, (self.pdict['num_reps'], 1, 1, 1))

        reg_strength = np.array([1e9])

        for i in range(0, len(reg_strength)):
            self.reg_strength = reg_strength[i]

            it_MSE, pol_SS = self.tune_model(self.pdict['it_lr'], self.pdict['it_epochs'], self.pdict, true_obs=true_obs,
                                             real_image=real_image, test_inputs_unnorm=test_inputs_unnorm,
                                             true_err=true_err,
                                             MCFOST=False, example_num=12 + i,
                                             tag='model_test_{}_regEWC{}_test'.format( self.pdict['images_path'], self.reg_strength))

