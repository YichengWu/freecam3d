'''
Main code to train the networks.
'''

import numpy as np
import def_dir
import tensorflow as tf
from tensorflow import ConfigProto
import os
from geometry import warp_p2c, z2pointcloud, bilinear_sampler, gen_visible_mask, coord0TOgrid1, pixel_to_ray_array, \
    points_in_camera_coords, zp_cView_to_zc
from data_provider import read_data
from network import UNet_xy, UNet_z
from utils import *
from optics import codePattern

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = ConfigProto()
config.gpu_options.allow_growth = True

######################################## Parameters ##################################################

# TODO: remove it for github version
DATA_PATH_root = def_dir.pattern_dir() + 'calibrationFree3D/Dataset/blender/1200_800/'
results_dir = def_dir.pattern_dir() + 'calibrationFree3D/tmp_0105_exp_psfs/'
pattern_type = 'kronTwoFix'
B = 1
B_sub = 10
N = 256
H = 800
W = 1200
N_layers = 21
weight_reProj = 1e0


def forward(DATA_PATH_root, B, B_sub, H, W, N_layers):
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

    # random crop to sub batches
    offset = tf.cast(tf.random.uniform([B_sub, 2]) * tf.constant([[H - N - 1, W - N - 1]], tf.float32), tf.int32)
    Ic_scaled_crop = multi_rand_crop(Ic_scaled, N, offset)
    Ic_mask_crop = multi_rand_crop(Ic_mask, N, offset)
    xyz_cView_crop = multi_rand_crop(xyz_cView, N, offset)
    xy_crop = multi_rand_crop(xy, N,
                              offset)  # this is just the relative xy value in an image (not for projector or camera)
    z_c_crop = multi_rand_crop(z_c, N, offset)
    Ic_crop = multi_rand_crop(Ic, N, offset)

    return Ic_scaled_crop, Ic_mask_crop, xyz_cView_crop, xy_crop, z_c_crop, pose_p2c, z_p, Ic_crop, Ip_coded


def recon(Ic_scaled_crop, Ic_mask_crop, xyz_cView_crop):
    if tf.contrib.framework.get_name_scope() == 'recon/train':
        phase_BN = True
    else:
        phase_BN = False

    Ic_scaled_crop = ln(Ic_scaled_crop, 17, 0.01)

    # xy/z two networks
    xy_cView_crop_hat = UNet_xy(Ic_scaled_crop, phase_BN)
    z_cView_crop_hat = UNet_z(Ic_scaled_crop, phase_BN)
    xyz_cView_crop_hat = tf.concat([xy_cView_crop_hat, z_cView_crop_hat], axis=-1)

    if tf.contrib.framework.get_name_scope() == 'recon/valid':
        tf.summary.image('Ic_scaled', Ic_scaled_crop[0:1, :, :, :])
        tf.summary.image('Ic_mask', Ic_mask_crop[0:1, :, :, :])
        tf.summary.image('x_cView', xyz_cView_crop[0:1, :, :, 0:1])
        tf.summary.image('x_cView_hat', xyz_cView_crop_hat[0:1, :, :, 0:1])
        tf.summary.image('y_cView', xyz_cView_crop[0:1, :, :, 1:2])
        tf.summary.image('y_cView_hat', xyz_cView_crop_hat[0:1, :, :, 1:2])
        z_min = 0.69
        tf.summary.image('z_cView', Ic_mask_crop[0:1, :, :, :] * (xyz_cView_crop[0:1, :, :, 2:3] - z_min))
        tf.summary.image('z_cView_hat', Ic_mask_crop[0:1, :, :, :] * (xyz_cView_crop_hat[0:1, :, :, 2:3] - z_min))

    return xyz_cView_crop_hat


def reProj_IpGT(zp_cView, pose_p2c, xy, Ic, zcGT, maskC, IpGT):
    maskC_expand = tf.cast((tf.nn.avg_pool(maskC, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')) > 0.999,
                           tf.float32)

    zp_cView = zp_cView * maskC_expand
    B = pose_p2c.shape[0].value
    B_all, N1, N2, _ = zp_cView.get_shape().as_list()
    B_sub = int(B_all / B)

    ## get grid_c2p
    zc = zp_cView_to_zc(zp_cView, pose_p2c, xy)
    coord_c = tf.transpose(
        points_in_camera_coords(zc, tf.concat([xy, tf.ones([B_all, N1, N2, 1])], axis=-1), allBatch=True), [0, 3, 1, 2])
    pose_c2p = tf.tile(tf.linalg.inv(pose_p2c), [B_sub, 1, 1])
    grid_c2p = coord0TOgrid1(coord_c, pose_c2p)

    ## warp Ip_Recoded to Ic
    Ic_reProj = bilinear_sampler(tf.tile(IpGT, [B_sub, 1, 1, 1]), grid_c2p)

    ## loss
    loss_zc_rms = cost_rms_mask(zcGT, zc, maskC_expand)
    loss_Ic_rms = cost_rms_mask(Ic, Ic_reProj, maskC_expand)
    loss_Ic_L1 = cost_L1_mask(Ic, Ic_reProj, maskC_expand)

    loss = loss_Ic_L1 + loss_zc_rms

    tf.summary.scalar('cost_zc_rms', loss_zc_rms)
    tf.summary.scalar('cost_Ic_rms', loss_Ic_rms)
    tf.summary.scalar('cost_Ic_L1', loss_Ic_L1)

    if tf.contrib.framework.get_name_scope() == 'reProj/valid':
        tf.summary.histogram('zc_hist', zc)
        tf.summary.histogram('zcGT_hist', zcGT)
        tf.summary.image('zc', 4 * (tf.clip_by_value(zc[0:1, :, :, :], 0.7, 0.95) - 0.7))
        tf.summary.image('zcGT', 4 * (tf.clip_by_value(zcGT[0:1, :, :, :], 0.7, 0.95) - 0.7))
        tf.summary.image('Ic_reProj', Ic_reProj[0:1, :, :, :])

    return loss


def cal_cost(xyz_cView_crop, xyz_cView_crop_hat, Ic_mask_crop):
    loss_xy_rms = cost_rms_mask(xyz_cView_crop[:, :, :, :2], xyz_cView_crop_hat[:, :, :, :2], Ic_mask_crop)
    loss_z_rms = cost_rms_mask(xyz_cView_crop[:, :, :, 2:3], xyz_cView_crop_hat[:, :, :, 2:3], Ic_mask_crop)
    Ic_mask_crop_expand = tf.cast(
        (tf.nn.avg_pool(Ic_mask_crop, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')) > 0.999, tf.float32)
    loss_z_grad = cost_grad_mask(xyz_cView_crop[:, :, :, 2:3], xyz_cView_crop_hat[:, :, :, 2:3], Ic_mask_crop_expand)

    tf.summary.scalar('cost_xy_rms', loss_xy_rms)
    tf.summary.scalar('cost_z_rms', loss_z_rms)
    tf.summary.scalar('cost_z_grad', loss_z_grad)

    ## combine loss from two parts
    loss = loss_xy_rms + loss_z_rms + loss_z_grad

    return loss


def main(_):
    ######################################## forward model ##################################################

    with tf.variable_scope("forward", reuse=tf.AUTO_REUSE):
        with tf.name_scope("train"):
            Ic_scaled_crop_train, Ic_mask_crop_train, xyz_cView_crop_train, xy_crop_train, zc_crop_train, pose_p2c_train, zp_train, Ic_crop_train, Ip_ref_train = forward(
                DATA_PATH_root, B, B_sub, H, W, N_layers)
        with tf.name_scope("valid"):
            Ic_scaled_crop_valid, Ic_mask_crop_valid, xyz_cView_crop_valid, xy_crop_valid, zc_crop_valid, pose_p2c_valid, zp_valid, Ic_crop_valid, Ip_ref_valid = forward(
                DATA_PATH_root, B, B_sub, H, W, N_layers)

    ######################################## Reconstruction ##################################################

    with tf.variable_scope('recon', reuse=tf.AUTO_REUSE):
        with tf.name_scope("train"):
            xyz_cView_crop_hat_train = recon(Ic_scaled_crop_train, Ic_mask_crop_train, xyz_cView_crop_train)
        with tf.name_scope("valid"):
            xyz_cView_crop_hat_valid = recon(Ic_scaled_crop_valid, Ic_mask_crop_valid, xyz_cView_crop_valid)

    ######################################## Loss ##################################################

    with tf.variable_scope('cost'):
        with tf.name_scope("train"):
            loss_train = cal_cost(xyz_cView_crop_train, xyz_cView_crop_hat_train, Ic_mask_crop_train)
        with tf.name_scope("valid"):
            loss_valid = cal_cost(xyz_cView_crop_valid, xyz_cView_crop_hat_valid, Ic_mask_crop_valid)

    ######################################## Reprojection Loss ##################################################

    with tf.variable_scope('reProj'):
        with tf.name_scope("train"):
            loss_reProj_train = reProj_IpGT(xyz_cView_crop_hat_train[:, :, :, 2:3], pose_p2c_train, xy_crop_train,
                                            Ic_crop_train, zc_crop_train, Ic_mask_crop_train, Ip_ref_train)
            loss_train = loss_train + loss_reProj_train * weight_reProj
        with tf.name_scope("valid"):
            loss_reProj_valid = reProj_IpGT(xyz_cView_crop_hat_valid[:, :, :, 2:3], pose_p2c_valid, xy_crop_valid,
                                            Ic_crop_valid, zc_crop_valid, Ic_mask_crop_valid, Ip_ref_valid)
            loss_valid = loss_valid + loss_reProj_valid * weight_reProj

    tf.summary.scalar('cost_all_train', loss_train)
    tf.summary.scalar('cost_all_valid', loss_valid)

    ######################################## Train setup ##################################################

    lr_val = 1e-4
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(lr_val).minimize(loss_train)

    merged = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=1)
    saver_best = tf.train.Saver()

    ######################################## Train  ##################################################

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        train_writer = tf.summary.FileWriter(results_dir + '/summary/', sess.graph)

        best_dir = 'best_model/'
        if not os.path.exists(results_dir + best_dir):
            os.makedirs(results_dir + best_dir)
            best_loss = 100
        else:
            best_loss = np.loadtxt(results_dir + 'best_loss.txt')
            print('Current best valid loss = ' + str(best_loss))

        if not tf.train.checkpoint_exists(results_dir + 'checkpoint'):
            print('Start to save at: ', results_dir)
        else:
            model_path = tf.train.latest_checkpoint(results_dir)
            saver.restore(sess, model_path)
            print('Continue to save at: ', results_dir)

        for i in range(1000000):
            train_op.run(feed_dict={})

            if i % 1000 == 0:
                [train_summary, loss_validt] = sess.run([merged, loss_valid],
                                                                    feed_dict={})

                print("Iter " + str(i) + ", Loss = " + "{:.6f}".format(loss_validt))

                train_writer.add_summary(train_summary, i)
                saver.save(sess, results_dir + "model.ckpt", global_step=i)

                if (loss_validt < best_loss) and (i > 1):
                    best_loss = loss_validt
                    np.savetxt(results_dir + 'best_loss.txt', [best_loss])
                    saver_best.save(sess, results_dir + best_dir + "model.ckpt")
                    print('best at iter ' + str(i) + ' with loss = ' + str(best_loss))

        coord.request_stop()


if __name__ == '__main__':
    tf.app.run()
