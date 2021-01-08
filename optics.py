'''
Function for optics related, mainly for pattern blur
'''

import tensorflow as tf
import numpy as np
import scipy.io as sio

#### optical parameters:
wvl = 530e-9
zernike = sio.loadmat('PSFs/zernike_basis75.mat')

# for Wm -60to60
z_min = 0.695714566
z_max = 0.941062289
z0 = 0.8
D = 21 / 2 * 1.4 * 1e-3

# These are the experimental calibrated PSFs.
fn_psfs = 'psfs_exp.npy'
PSFs_np = np.load(fn_psfs)


def gen_WmMask(WmMap, Wm_layers):
    # generate binary mask for different Wm
    N_layers = len(Wm_layers)
    Wm_step = (Wm_layers[-1] - Wm_layers[0]) / (N_layers - 1)  # find the step size of Wm_layers

    # convert to integer step size
    Wm_layers_normalized = Wm_layers / Wm_step
    WmMap_normalized = WmMap / Wm_step

    mask = tf.zeros([WmMap.shape[0].value, WmMap.shape[1].value, WmMap.shape[2].value, 0])
    for Wm in Wm_layers_normalized:
        mask0 = tf.cast(tf.abs(tf.round(WmMap_normalized) - Wm) < 0.01, tf.float32)
        mask = tf.concat([mask, mask0], axis=3)
    return mask


def z2Wm(z):
    k = 2 * np.pi / wvl
    Wm = k * (D / 2) ** 2 / 2 * (1 / z - 1 / z0)
    return Wm



def codePattern(pattern, zMap):
    N_layers = pattern.shape[-1].value
    Wm_min = z2Wm(z_max)
    Wm_max = z2Wm(z_min)
    Wm_layers = np.linspace(Wm_min, Wm_max, N_layers)

    WmMap = z2Wm(zMap)
    WmMask = gen_WmMask(WmMap, Wm_layers)  # binary mask to see which layer each point locates

    PSFs = tf.cast(PSFs_np, tf.float32)

    Ip_coded = tf.reduce_sum(tf.nn.conv2d(pattern[:, :, :, 0:1], tf.transpose(PSFs, [1, 2, 3, 0]), strides=[1, 1, 1, 1],
                                          padding='SAME') * WmMask, axis=-1, keepdims=True)

    # normalized to max = 1
    norm = tf.reduce_max(tf.reduce_max(Ip_coded, 1, keep_dims=True), 2, keep_dims=True)
    Ip_coded_norm = Ip_coded / (norm + 1e-10)

    return Ip_coded_norm
