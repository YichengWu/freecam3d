import os
import tensorflow as tf
import numpy as np
import scipy.io as sio


##########################################   generate grid pattern on the projector  #############################################

def gen_pattern(B, H, W, N_layers, grid_Isigma, patternMode, stride):
    grid_intensity = tf.nn.relu(tf.random_normal([B, 1, 1, 1], 1, grid_Isigma))
    if patternMode == "grid":
        grid0 = np.zeros([B, H + 2 * stride, W + 2 * stride, N_layers], np.float32)
        grid0[:, 0: -1: stride, :, :] = 1
        grid0[:, :, 0: -1: stride, :] = 1
        grid1 = tf.image.random_crop(grid0, [B, H, W, N_layers])
    if patternMode == "kinect":
        kinect_mat = sio.loadmat('ProjPatterns/kinect1200.mat')
        kinect_np = kinect_mat['binary']
        grid1 = tf.tile(tf.reshape(tf.constant(kinect_np, tf.float32), [1, H, W, 1]), [B, 1, 1, N_layers])
    if patternMode == "MArray":
        kinect_mat = sio.loadmat('ProjPatterns/M_array.mat')
        kinect_np = kinect_mat['squares']
        grid1 = tf.tile(tf.reshape(tf.constant(kinect_np, tf.float32), [1, H, W, 1]), [B, 1, 1, N_layers])
    if patternMode == "kronTwoFix":
        ## local pattern
        # cross
        Pla = np.zeros([stride, stride, 1, 1], np.float32)
        Pla[int(stride / 2), :, :, :] = 1
        Pla[:, int(stride / 2), :, :] = 1
        Pla = tf.cast(Pla, tf.float32)

        # square
        Plb = np.zeros([stride, stride, 1, 1], np.float32)
        Plb[int(stride / 4), int(stride / 4):int(3 * stride / 4), :, :] = 1
        Plb[int(3 * stride / 4), int(stride / 4):int(3 * stride / 4), :, :] = 1
        Plb[int(stride / 4):int(3 * stride / 4), int(stride / 4), :, :] = 1
        Plb[int(stride / 4):int(3 * stride / 4) + 1, int(3 * stride / 4), :, :] = 1

        ## global pattern
        if stride == 16:
            Pg = tf.cast(np.load('ProjPatterns/kron_Pg_50x75_0.5.npy'), tf.float32)
        else:
            raise ValueError('the stride has to be 16 for kronTwoFix.')

        grid0a = tf.nn.conv2d_transpose(Pg, Pla, [1, H, W, 1], [1, stride, stride, 1], "VALID")
        grid0b = tf.nn.conv2d_transpose(1 - Pg, Plb, [1, H, W, 1], [1, stride, stride, 1], "VALID")

        grid1 = tf.tile(grid0a + grid0b, [B, 1, 1, N_layers])


    grid = tf.multiply(grid1, grid_intensity)
    return grid


#######################################  loss function  #############################################

def cost_rms_mask(GT, hat, mask):
    loss = tf.sqrt(tf.reduce_sum(tf.square(mask * (GT - hat))) / (tf.reduce_sum(mask) + 1))
    return loss


def cost_L1_mask(GT, hat, mask):
    loss = tf.reduce_sum(mask * tf.abs(GT - hat)) / (tf.reduce_sum(mask) + 1)
    # loss = tf.reduce_mean(mask * tf.abs(GT - hat))
    return loss


def cost_grad_mask(GT, hat, mask):
    mask_expand = tf.cast((tf.nn.avg_pool(mask, ksize=[1, 9, 9, 1], strides=[1, 1, 1, 1], padding='SAME')) > 0.999,
                          tf.float32)
    [GTy, GTx] = tf.image.image_gradients(GT)
    [haty, hatx] = tf.image.image_gradients(hat)
    costx = cost_rms_mask(GTx, hatx, mask_expand)
    costy = cost_rms_mask(GTy, haty, mask_expand)

    return costx + costy


#####################################   crop image to multiple random patches  ################################

def multi_rand_crop(I, Ic_size, offset):
    _, H, W, C = I.get_shape().as_list()
    Ic = tf.zeros([0, Ic_size, Ic_size, C])
    for i in range(offset.shape[0].value):
        Ic0 = tf.image.crop_to_bounding_box(I, offset[i, 0], offset[i, 1], Ic_size, Ic_size)
        Ic = tf.concat([Ic, Ic0], axis=0)

    return Ic


#####################################   adjust the pattern intensity based on the reflectance  ################################

def depth2SN(depth, pixelsize=200e-6):
    # get surface normal from depth
    [dy, dx] = tf.image.image_gradients(depth)
    dM = tf.sqrt(dy ** 2 + dx ** 2)
    tan = dM / pixelsize
    theta = tf.atan(tan)
    return theta


def withReflectance(I, depth):
    # change the intensity of the pattern based on the relfectance
    SN = depth2SN(depth, 100e-6)
    reflectance = tf.cos(SN) * 0.8 + 0.2  # add a DC and cosine term
    I_ref = I * reflectance

    return I_ref


##########################################   local normalization  #############################################

def ln(Im, kernelsize, constant):
    mean = tf.nn.avg_pool(Im, [1, kernelsize, kernelsize, 1], [1, 1, 1, 1], 'SAME')
    norm = Im / (mean + constant)
    return norm


##########################################   add random texture from the scene  #############################################

def addTexture(Ip_coded, variation):
    [B, H, W, C] = Ip_coded.get_shape().as_list()

    # create a small random pattern and resize it
    base_shape = tf.cast(tf.random.uniform([2]) * 18 + 2, tf.int32)
    texture0 = tf.random.uniform(base_shape) * variation + (1 - variation)  # the intensty range 1-variation to 1
    texture1 = tf.tile(tf.expand_dims(tf.expand_dims(tf.cast(texture0, tf.float32), 0), -1), [B, 1, 1, C])

    # scale up using different interpolation
    texture2 = tf.cond(tf.random.uniform([])>0.5,
                       lambda: tf.image.resize(texture1, [W, W], method = tf.image.ResizeMethod.BILINEAR),
                       lambda: tf.image.resize(texture1, [W, W], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR))


    texture3 = tf.image.random_crop(texture2, [B, H, W, C])

    return Ip_coded * texture3

def addTexture_checkerboard(Ip_coded, contrast):
    [B, H, W, C] = Ip_coded.get_shape().as_list()

    # create a small pattern and resize it
    size = 5
    texture0 = np.ones((B, size, size, C), dtype=int)/contrast   # two intensities: 1/contrast and 1
    texture0[:, 1::2, ::2, :] = 1 # variation
    texture0[:, ::2, 1::2, :] = 1 # variation
    texture1 = tf.cast(texture0, tf.float32)

    # scale up
    texture2 = tf.image.resize(texture1, [W, W], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    texture3 = tf.image.crop_to_bounding_box(texture2, 0, 0, H, W)

    return Ip_coded * texture3