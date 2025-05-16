

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from final_imrfitnn_jatis import imrnn
import os
import shutil
from tensorflow import keras
import sys


model_name =  'model_paper'
it_fit_eg_path = 'mcfost_test_1.npy' # supplied simulated astrophysical images for iterative fitting
datadir = '../image_recon_data/'


# build your network design here
pdict = {}

# data config
pdict['batchSize'] =  32
pdict['num_train'] = 8200
pdict['num_test'] = 200

# network architecture config
pdict['kernel_size'] = 4
pdict['num_kern_first'] = 200
pdict['num_kern'] = 200
pdict['stride_layers'] = 5
pdict['reshape_spatial'] = 4
pdict['reshape_channel'] = 256
pdict['n_lay'] = 1

# network training and iterative fitting config
pdict['learningRate'] = 1e-5
pdict['epochs'] =  20 #000
pdict['it_epochs'] = 100#0000
pdict['it_lr'] = 1e-6
pdict['num_reps'] = 75
pdict['dropout_rate'] = 0.3
pdict['leaky_relu'] = 0.25


pdict['images_path'] = it_fit_eg_path


imr = imrnn(datadir, model_name)
imr.pdict = pdict
imr.load_training_data(datadir)
imr.build_model_transpose_layers()
imr.train_model(pdict, tag = 'testing_network_', showplots = True)
val_image_mse, train_image_mse, image_tv, entropy, valssim = imr.show_predictions(subtitle='testing_network_', tag='testing_network_')
imr.iterative_fitting_example( pdict['images_path'])


