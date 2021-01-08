'''
Code to read data from tfrecord, mainly the data I generated from Blender
'''


import tensorflow as tf
import glob



def parse_element(example):
    # from tfrecord file to data (uint16)
    H = 800
    W = 1200
    features = tf.parse_single_example(example,
                                       features={
                                           'z_p': tf.FixedLenFeature([], tf.string),
                                           'z_c': tf.FixedLenFeature([], tf.string),
                                           'pose_p2c': tf.FixedLenFeature([4 * 4, ], tf.float32)
                                       })
    z_p_flat = tf.cast(tf.decode_raw(features['z_p'], tf.uint16), tf.float32) / 50000 + 0.3
    z_c_flat = tf.cast(tf.decode_raw(features['z_c'], tf.uint16), tf.float32) / 50000 + 0.3
    pose_p2c_flat = features['pose_p2c']
    z_p = tf.reshape(z_p_flat, [H, W, 1])
    z_c = tf.reshape(z_c_flat, [H, W, 1])
    pose_p2c = tf.reshape(pose_p2c_flat, [4, 4])

    return z_p, z_c, pose_p2c




def read_data(root_path, batchsize):
    isTrain = True
    if tf.contrib.framework.get_name_scope() == 'forward/train':
        path = glob.glob(root_path + 'train*.tfrecord')
    elif tf.contrib.framework.get_name_scope() == 'forward/valid':
        path = glob.glob(root_path + 'valid*.tfrecord')
    else:
        path = glob.glob(root_path + 'test.tfrecord')
        isTrain = False

    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(parse_element, num_parallel_calls=8)
    if isTrain:
        dataset = dataset.shuffle(buffer_size=200).repeat()
    dataset = dataset.batch(batchsize, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=100)
    iterator = dataset.make_one_shot_iterator()
    z_p, z_c, pose_p2c = iterator.get_next()

    return z_p, z_c, pose_p2c