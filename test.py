'''
test code for simulation data
'''

import numpy as np
import tensorflow as tf
from tensorflow import ConfigProto
import os
from geometry import warp_p2c, z2pointcloud, bilinear_sampler, gen_visible_mask, pixel_to_ray_array
from data_provider import read_data
from network import UNet_xy, UNet_z
from utils import *
from optics import codePattern
import imageio
import matplotlib.image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = ConfigProto()
config.gpu_options.allow_growth = True

######################################## Parameters ##################################################

DATA_PATH_root = './Dataset/'
results_dir = './FreeCam3D_model/'
out_dir = 'test/'
pattern_type = 'kronTwoFix'
NN = 10
B = 1
H = 800
W = 1200
N_layers = 21


def forward(DATA_PATH_root, B, H, W, N_layers):
    z_p, z_c, pose_p2c = read_data(DATA_PATH_root, B)

    # convert depthmap to 3D point cloud
    coord_p = z2pointcloud(z_p)
    coord_c = z2pointcloud(z_c)

    Ip = gen_pattern(B, H, W, N_layers, 0, pattern_type, 16)
    Ip_ref = withReflectance(Ip, z_p)

    ## generate coded projector pattern
    Ip_coded = codePattern(Ip_ref, z_p)

    Ic, grid_p2c, grid_c2p = warp_p2c(Ip_coded, coord_p, coord_c, pose_p2c)
    Ic_exp = Ic + tf.random.uniform([B, 1, 1, 1], maxval=0.05) + tf.random.normal([B, H, W, 1],
                                                                                  stddev=0.005)  # add bg and noise
    Ic_scaled = Ic_exp / tf.reduce_max(tf.reduce_max(Ic_exp, 1, keep_dims=True), 2,
                                       keep_dims=True) * tf.random.uniform([B, 1, 1, 1], minval=0.7,
                                                                           maxval=1)  # scale so that max is 0.7-1
    Ic_scaled = tf.clip_by_value(Ic_scaled, 0.0, 1.0)

    # get world coordinate x,y,z in camera view
    visible_mask_c_dense = gen_visible_mask(grid_p2c, 'all')

    xy = tf.cast(tf.tile(tf.expand_dims(pixel_to_ray_array()[:, :, 0:2], 0), [B, 1, 1, 1]),
                 tf.float32)  # this is x/z and y/z
    xyz = tf.concat([xy, z_p], axis=-1)
    xyz_cView = bilinear_sampler(xyz, grid_c2p) * visible_mask_c_dense

    Ic_mask = tf.cast(Ic_scaled > 0.05, tf.float32) * visible_mask_c_dense  # Ic mask

    return Ic_scaled, Ic_mask, xyz_cView, z_c, pose_p2c, z_p, Ic, Ip_coded, Ip


def recon(Ic_scaled, Ic_mask, xyz_cView):
    if tf.contrib.framework.get_name_scope() == 'recon/train':
        phase_BN = True
    else:
        phase_BN = False

    Ic_scaled = ln(Ic_scaled, 17, 0.01)

    # xy/z two networks
    xy_cView_crop_hat = UNet_xy(Ic_scaled, phase_BN)
    z_cView_crop_hat = UNet_z(Ic_scaled, phase_BN)
    xyz_cView_hat = tf.concat([xy_cView_crop_hat, z_cView_crop_hat], axis=-1)

    if tf.contrib.framework.get_name_scope() == 'recon/valid':
        tf.summary.image('Ic_scaled', Ic_scaled[0:1, :, :, :])
        tf.summary.image('Ic_mask', Ic_mask[0:1, :, :, :])
        tf.summary.image('x_cView', xyz_cView[0:1, :, :, 0:1])
        tf.summary.image('x_cView_hat', xyz_cView_hat[0:1, :, :, 0:1])
        tf.summary.image('y_cView', xyz_cView[0:1, :, :, 1:2])
        tf.summary.image('y_cView_hat', xyz_cView_hat[0:1, :, :, 1:2])
        z_min = 0.69
        tf.summary.image('z_cView', Ic_mask[0:1, :, :, :] * (xyz_cView[0:1, :, :, 2:3] - z_min))
        tf.summary.image('z_cView_hat', Ic_mask[0:1, :, :, :] * (xyz_cView_hat[0:1, :, :, 2:3] - z_min))

    return xyz_cView_hat, Ic_scaled

def cal_cost(xyz_cView, xyz_cView_hat, Ic_mask):
    loss_y_rms = cost_rms_mask(xyz_cView[:, :, :, 1:2]*xyz_cView[:, :, :, 2:3], xyz_cView_hat[:, :, :, 1:2]*xyz_cView[:, :, :, 2:3], Ic_mask)
    loss_x_rms = cost_rms_mask(xyz_cView[:, :, :, 0:1]*xyz_cView[:, :, :, 2:3], xyz_cView_hat[:, :, :, 0:1]*xyz_cView[:, :, :, 2:3], Ic_mask)

    loss_z_rms = cost_rms_mask(xyz_cView[:, :, :, 2:3], xyz_cView_hat[:, :, :, 2:3], Ic_mask)

    return loss_x_rms,loss_y_rms,loss_z_rms


def main(_):
    ######################################## forward model ##################################################

    with tf.variable_scope("forward", reuse=tf.AUTO_REUSE):

        with tf.name_scope("test"):
            Ic_scaled, Ic_mask, xyz_cView, zc_crop, pose_p2c, zp, Ic, Ip_coded, Ip = forward(
                DATA_PATH_root, B, H, W, N_layers)

    ######################################## Reconstruction ##################################################

    with tf.variable_scope('recon', reuse=tf.AUTO_REUSE):
        with tf.name_scope("test"):
            xyz_cView_hat, Ic_coded_LN = recon(Ic_scaled, Ic_mask, xyz_cView)


    ######################################## Loss ##################################################

    with tf.variable_scope('cost'):
        with tf.name_scope("test"):
            loss_x_rms,loss_y_rms,loss_z_rms = cal_cost(xyz_cView, xyz_cView_hat, Ic_mask)


    ######################################## Test  ##################################################
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()

        if not os.path.exists(results_dir + out_dir):
            os.makedirs(results_dir + out_dir)

        model_path = tf.train.latest_checkpoint(results_dir)
        saver.restore(sess, model_path)
        print('Load at: ', results_dir)

        loss_x_rms_all = []
        loss_y_rms_all = []
        loss_z_rms_all = []

        for j in range(NN):

            [xyz_cView_hatt, Ic_scaledt, xyz_cViewt, pose_p2ct, Ip_codedt, Ipt, Ic_coded_LNt, loss_x_rmst,loss_y_rmst,loss_z_rmst, zpt] = sess.run(
                [xyz_cView_hat, Ic_scaled, xyz_cView, pose_p2c, Ip_coded, Ip, Ic_coded_LN, loss_x_rms,loss_y_rms,loss_z_rms, zp])
            print(j)
            print(loss_x_rmst)
            print(loss_y_rmst)
            print(loss_z_rmst)

            loss_x_rms_all.append(loss_x_rmst)
            loss_y_rms_all.append(loss_y_rmst)
            loss_z_rms_all.append(loss_z_rmst)

            np.savetxt(results_dir + out_dir + '%04d_pose_p2cGT.txt' % (j),pose_p2ct[0,:,:])

            imageio.imwrite(results_dir + out_dir + '%04d_xHat.png' % (j),
                            np.uint8((xyz_cView_hatt[0, :, :, 0] + 0.36) / (0.36 * 2) * 255))
            imageio.imwrite(results_dir + out_dir + '%04d_xGT.png' % (j),
                            np.uint8((xyz_cViewt[0, :, :, 0] + 0.36) / (0.36 * 2) * 255))

            imageio.imwrite(results_dir + out_dir + '%04d_yHat.png' % (j),
                            np.uint8((xyz_cView_hatt[0, :, :, 1] + 0.36) / (0.36 * 2) * 255))
            imageio.imwrite(results_dir + out_dir + '%04d_yGT.png' % (j),
                            np.uint8((xyz_cViewt[0, :, :, 1] + 0.36) / (0.36 * 2) * 255))

            imageio.imwrite(results_dir + out_dir + '%04d_zHat.png' % (j),
                            np.uint16((xyz_cView_hatt[0, :, :, 2] - 0.7) * 240000))
            imageio.imwrite(results_dir + out_dir + '%04d_zGT.png' % (j),
                            np.uint16((xyz_cViewt[0, :, :, 2] - 0.7) * 240000))
            imageio.imwrite(results_dir + out_dir + '%04d_zpGT.png' % (j),
                            np.uint16((zpt[0, :, :, 0] - 0.7) * 240000))

            matplotlib.image.imsave(results_dir + out_dir + '%04d_zjetHat.png' % (j),
                                    xyz_cView_hatt[0, :, :, 2], vmin=0.7, vmax=0.94)
            matplotlib.image.imsave(results_dir + out_dir + '%04d_zjetGT.png' % (j),
                                    xyz_cViewt[0, :, :, 2], vmin=0.7, vmax=0.94)
            matplotlib.image.imsave(results_dir + out_dir + '%04d_zpjetGT.png' % (j),
                                    zpt[0, :, :, 0], vmin=0.7, vmax=0.94)

            imageio.imwrite(results_dir + out_dir + '%04d_Ic.png' % (j),
                            np.uint8(Ic_scaledt[0, :, :, 0] * 255))

            imageio.imwrite(results_dir + out_dir + '%04d_IcLN.png' % (j),
                            np.uint8(Ic_coded_LNt[0, :, :, 0] * 255 / np.max(Ic_coded_LNt[0, :, :, 0])))

            imageio.imwrite(results_dir + out_dir + '%04d_Ip_coded.png' % (j),
                            np.uint8(Ip_codedt[0, :, :, 0] * 255))
            imageio.imwrite(results_dir + out_dir + '%04d_Ip.png' % (j),
                            np.uint8(Ipt[0, :, :, 0] * 255))

        loss_x_rms_avg = np.mean(loss_x_rms_all)
        loss_y_rms_avg = np.mean(loss_y_rms_all)
        loss_z_rms_avg = np.mean(loss_z_rms_all)
        print('lossX='+str(loss_x_rms_avg))
        print('lossY='+str(loss_y_rms_avg))
        print('lossZ='+str(loss_z_rms_avg))

        np.savetxt(results_dir + out_dir + 'lossX_avg=%.4f.txt' % loss_x_rms_avg, loss_x_rms_all)
        np.savetxt(results_dir + out_dir + 'lossY_avg=%.4f.txt' % loss_y_rms_avg, loss_y_rms_all)
        np.savetxt(results_dir + out_dir + 'lossZ_avg=%.4f.txt' % loss_z_rms_avg, loss_z_rms_all)

        coord.request_stop()


if __name__ == '__main__':
    tf.app.run()
