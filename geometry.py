'''
Geometric functions. Parameters are for Blender simulation images with 800x1200 resolution
'''

import numpy as np
import tensorflow as tf
import math

H = 800
W = 1200


def pixel_to_ray(pixel, pixel_width=1200, pixel_height=800):
    x, y = pixel
    x_vect = pixel_width / (2*1666.67) * ((2.0 * ((x + 0.5) / pixel_width)) - 1.0)
    y_vect = pixel_height / (2*1666.67) * ((2.0 * ((y + 0.5) / pixel_height)) - 1.0)
    return (x_vect, y_vect, 1.0)


def pixel_to_ray_array(width=1200, height=800):
    # for this un-normalized version, final ray z = 1
    pixel_to_ray_array = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            pixel_to_ray_array[y, x] = np.array(pixel_to_ray((x, y), pixel_height=height, pixel_width=width))
    return pixel_to_ray_array


def points_in_camera_coords(z_map, pixel_to_ray_array, allBatch = False):
    # this z map is a scaling factor of the ray
    B, H, W, _ = z_map.get_shape().as_list()
    z_map3 = tf.tile(z_map, [1, 1, 1, 3])
    if allBatch:  # I don't need to tile to all batch
        pixel_to_ray_array_batch = tf.cast(pixel_to_ray_array, tf.float32)
    else:
        pixel_to_ray_array_batch = tf.tile(tf.expand_dims(tf.cast(pixel_to_ray_array, tf.float32), 0), [B, 1, 1, 1])
    camera_relative_xyz = z_map3 * pixel_to_ray_array_batch
    ones = tf.ones([B, H, W, 1])
    return tf.concat([camera_relative_xyz, ones], axis=-1)


def coord_transform(coord, transform):
    """Transforms 4D coordinate to 2D pixel coordinate based on the transformation matrix.

    Args:
      coord: [B, 4, H, W]
      transform: [B, 4, 4]
    Returns:
      2D pixel coordinate, shape = [B, H, W, 2].
    """
    B, _, H, W = coord.get_shape().as_list()
    coord = tf.reshape(coord, [B, 4, -1])
    unnormalized_pixel_coords = tf.matmul(transform, coord)
    x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
    y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
    z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
    pixel_coords = tf.concat([x_n, y_n], axis=1)
    pixel_coords = tf.reshape(pixel_coords, [B, 2, H, W])
    return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])


def coord0TOgrid1(coord0, pose01):
    """Project 4D space coordinate in view 0 to pixel coordinate in view 1.

    Args:
        coord0: 4D space coordinate in view 0, shape = [B, 4, H, W].
        pose01: relative pose tranformation from view 0 to view 1, shape = [B, 4, 4].

    Returns:
        grid1: the 2D pixel coordinate in view 1
    """
    B = coord0.shape[0].value
    intrinsics = get_intrinsic(B)
    transform = tf.matmul(intrinsics, pose01)
    grid1 = coord_transform(coord0, transform)
    return grid1


def bilinear_sampler(imgs, coords):
    """Construct a new image by bilinear sampling from the input image.

    Points falling outside the source image boundary have value 0.
    Function from https://github.com/tinghuiz/SfMLearner.

    Args:
      imgs: source image to be sampled from [batch, height_s, width_s, channels]
      coords: coordinates of source pixels to sample from [batch, height_t,
        width_t, 2]. height_t/width_t correspond to the dimensions of the output
        image (don't need to be the same as height_s/width_s). The two channels
        correspond to x and y coordinates respectively.
    Returns:
      A new sampled image [batch, height_t, width_t, channels]
    """

    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
            ])), 1), [1, 0])
        rep = tf.cast(rep, 'float32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    with tf.name_scope('image_sampling'):
        coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
        inp_size = imgs.get_shape()
        coord_size = coords.get_shape()
        out_size = coords.get_shape().as_list()
        out_size[3] = imgs.get_shape().as_list()[3]

        coords_x = tf.cast(coords_x, 'float32')
        coords_y = tf.cast(coords_y, 'float32')

        x0 = tf.floor(coords_x)
        x1 = x0 + 1
        y0 = tf.floor(coords_y)
        y1 = y0 + 1

        y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
        x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
        zero = tf.zeros([1], dtype='float32')

        x0_safe = tf.clip_by_value(x0, zero, x_max)
        y0_safe = tf.clip_by_value(y0, zero, y_max)
        x1_safe = tf.clip_by_value(x1, zero, x_max)
        y1_safe = tf.clip_by_value(y1, zero, y_max)

        wt_x0 = x1_safe - coords_x
        wt_x1 = coords_x - x0_safe
        wt_y0 = y1_safe - coords_y
        wt_y1 = coords_y - y0_safe

        ## indices in the flat image to sample from
        dim2 = tf.cast(inp_size[2], 'float32')
        dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
        base = tf.reshape(
            _repeat(
                tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
                coord_size[1] * coord_size[2]),
            [out_size[0], out_size[1], out_size[2], 1])

        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = tf.reshape(x0_safe + base_y0, [-1])
        idx01 = x0_safe + base_y1
        idx10 = x1_safe + base_y0
        idx11 = x1_safe + base_y1

        ## sample from imgs
        imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
        imgs_flat = tf.cast(imgs_flat, 'float32')
        im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
        im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
        im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
        im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        output = tf.add_n([
            w00 * im00, w01 * im01,
            w10 * im10, w11 * im11
        ])
        return output



def gen_visible_mask(grid, method='round'):
    """Creates a visible mask by (forward) warping.

    Varible `grid` provides many 2D locatiions, mark them (nearest grid) as visible.

    Args:
        grid: 2D pixel coordinate, shape = [B, H, W, 2]
        method: "all" or "round". "all" means to mark 1 as many as possible. "round" means to mark 1 as few as possible.

    Returns:
        visible_mask: a 2D visible mask, shape = [B, H, W, 1]
    """
    B = grid.shape[0].value
    H = grid.shape[1].value
    W = grid.shape[2].value

    if method == 'all':
        grid_idx_ff = tf.floor(grid[:, :, :, 0]) + tf.floor(grid[:, :, :, 1]) * W
        grid_idx_fc = tf.floor(grid[:, :, :, 0]) + tf.ceil(grid[:, :, :, 1]) * W
        grid_idx_cf = tf.ceil(grid[:, :, :, 0]) + tf.floor(grid[:, :, :, 1]) * W
        grid_idx_cc = tf.ceil(grid[:, :, :, 0]) + tf.ceil(grid[:, :, :, 1]) * W

        grid_idx = tf.stack([grid_idx_ff, grid_idx_fc, grid_idx_cf, grid_idx_cc], axis=3)

    elif method == 'round':
        grid_idx = tf.round(grid[:, :, :, 0]) + tf.round(grid[:, :, :, 1]) * W

    grid_idx_1d = tf.cast(tf.reshape(grid_idx, [B, -1]), tf.int32)
    ones = tf.ones_like(grid_idx_1d)
    mask = tf.zeros([0, H * W], tf.bool)
    for i in range(B):
        m = tf.expand_dims(tf.scatter_nd(tf.expand_dims(grid_idx_1d[i, :], -1), ones[i, :], tf.constant([H * W])) > 0,
                           0)
        mask = tf.concat([mask, m], axis=0)

    visible_mask = tf.cast(tf.reshape(mask, [B, H, W, 1]), tf.float32)

    return visible_mask


def z2pointcloud(z):
    """Converts zMap (the actual value in 3D point cloud) to 4D coordinate (x, y, z ,1)

    Args:
        z: z map in Cartesian coordinate, shape = [B, H, W, 1]

    Returns:
        coord: 4D coordinate, shape = [B, 4, H, W]
    """
    # covert
    cached_pixel_to_ray_array = pixel_to_ray_array()
    coord = tf.transpose(points_in_camera_coords(z, cached_pixel_to_ray_array), [0, 3, 1, 2])
    return coord


def get_intrinsic(B):
    # intrinsic info for the sceneNet dataset
    Intrinsic = np.diag([1666.67, 1666.67, 1, 1])
    Intrinsic[0][2] = 1200/2
    Intrinsic[1][2] = 800/2
    Intrinsic = tf.cast(Intrinsic, tf.float32)
    Intrinsic = tf.tile(tf.cast(tf.expand_dims(Intrinsic, 0), tf.float32), [B, 1, 1])  # (B,4,4)

    return Intrinsic

def warp_p2c(Ip, coord_p, coord_c, pose_p2c):
    """ Warps the image Ip (projector view) to Ic (camera view).

    Args:
        Ip: Image from the projector view, shape = [B, H, W, 1].
        coord_p: 4D scene coordinate in projector view, shape = [B, 4, H, W].
        coord_c: 4D scene coordinate in camera view, shape = [B, 4, H, W].
        pose_p2c: camera pose transformation from projector to camera, shape = [B, 4, 4].

    Returns:
        Ic_masked: Warped image with occluded regions to be black, shape = [B, H, W, 1].
        grid_p2c: For coord_p, the corresponding 2D pixel coordinate in camera view, shape = [B, H, W, 2].
        grid_c2p: For coord_c, the corresponding 2D pixel coordinate in projector view, shape = [B, H, W, 2].
    """

    # inverse warp to get intensity in camera view
    pose_c2p = tf.linalg.inv(pose_p2c)
    grid_c2p = coord0TOgrid1(coord_c, pose_c2p)
    Ic = bilinear_sampler(Ip, grid_c2p)  # inverse warp

    # forward warp to get mask in camera view
    grid_p2c = coord0TOgrid1(coord_p, pose_p2c)
    visible_mask_c = gen_visible_mask(grid_p2c, 'all')

    Ic_masked = Ic * visible_mask_c

    return Ic_masked, grid_p2c, grid_c2p


def zp_cView_to_zc(zp_cView, pose_p2c, xy):
    # from zp in camera view to zc in camera view
    pose_c2p = tf.linalg.inv(pose_p2c)
    a20 = pose_c2p[0,2,0]
    a21 = pose_c2p[0,2,1]
    a22 = pose_c2p[0,2,2]
    a23 = pose_c2p[0,2,3]

    zc = (zp_cView - a23) / (a20 * xy[:, :, :, 0:1] + a21 * xy[:, :, :, 1:2] + a22)

    return zc
