from functools import partial
import numpy as np
import slim
import tensorflow as tf

import utils
from tensorflow.python.framework import ops as op
from slim import ops
from slim import scopes


op.NotDifferentiable("ExtractPatches")


def align_reference_shape(reference_shape, reference_shape_bb, im, bb):
    def norm(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x - tf.reduce_mean(x, 0))))

    ratio = norm(bb) / norm(reference_shape_bb)
    align_mean_shape = (reference_shape - tf.reduce_mean(reference_shape_bb, 0)) * ratio + tf.reduce_mean(bb, 0)
    new_size = tf.to_int32(tf.to_float(tf.shape(im)[:2]) / ratio)
    return tf.image.resize_bilinear(tf.expand_dims(im, 0), new_size)[0, :, :, :], align_mean_shape / ratio, ratio


def normalized_rmse_aflw(pred, gt_truth):
    norm = np.sqrt(np.sum(((gt_truth[:, 6, :] - gt_truth[:, 11, :]) ** 2), 1))  # out-ocular distance
    # print('norm:', norm)
    return np.sum(np.sqrt(np.sum(np.square(pred - gt_truth), 2)), 1) / (norm * 19)


def normalized_rmse(pred, gt_truth):
    norm = np.sqrt(np.sum(((gt_truth[:, 36, :] - gt_truth[:, 45, :]) ** 2), 1))  # out-ocular distance
    # print('norm:', norm)
    return np.sum(np.sqrt(np.sum(np.square(pred - gt_truth), 2)), 1) / (norm * 68)


def normalized_rmse_2(pred, gt_truth, bbox):
    norm = tf.cast(tf.sqrt(((bbox[:, 2, 1] - bbox[:, 0, 1]) * (bbox[:, 2, 0] - bbox[:, 0, 0]))),
                   tf.float32)  # out-ocular distance

    return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(pred - gt_truth), 2)), 1) / (norm * 19)


def pad_patches(images, patch_shape, shapes):
    batch_size = images.get_shape().as_list()[0]
    # pad_output = tf.zeros((k, knn_2D.get_shape().as_list()[1:]))
    m_module = tf.load_op_library('extract_patches.so')
    pad_output = m_module.extract_patches(images, tf.constant(patch_shape), shapes)
    return pad_output  # k, num_patches, height, width, 3


def step(sess, a, s, gt):
    # FUCK! DO NOT USE sess.run()!
    # gt_tf = tf.convert_to_tensor(gt, dtype=tf.float32)
    # pred = tf.convert_to_tensor(a, dtype=tf.float32)
    norm = np.sqrt(np.sum(((gt[:, 36, :] - gt[:, 45, :]) ** 2), 1))
    # error_s = np.sqrt(np.sum(np.square(s - gt), 2)) / norm
    error_s = np.sum(np.sqrt(np.sum(np.square(s - gt), 2)), 1) / (norm * 68)
    error_a = np.sum(np.sqrt(np.sum(np.square(s + a - gt), 2)), 1) / (norm * 68)
    reward = (error_s - error_a)
    # print('reward:', reward.shape)
    # print('error_s:', error_s)
    # print('error_a:', error_a)
    # print np.sign(error_s - error_a).reshape([-1, 68, 1])
    return s + a, reward



def conv_model(inputs, is_training=True, scope=''):
    # summaries or losses.
    net = {}

    with tf.name_scope(scope, 'rdn_conv', [inputs]):
        with scopes.arg_scope([ops.conv2d, ops.fc], is_training=is_training):
            with scopes.arg_scope([ops.conv2d], activation=tf.nn.relu, padding='VALID'):
                net['conv_1'] = ops.conv2d(inputs, 32, [3, 3], scope='conv_1')
                net['pool_1'] = ops.max_pool(net['conv_1'], [2, 2])
                net['conv_2'] = ops.conv2d(net['pool_1'], 32, [3, 3], scope='conv_2')
                net['pool_2'] = ops.max_pool(net['conv_2'], [2, 2])

                crop_size = net['pool_2'].get_shape().as_list()[1:3]
                net['conv_2_cropped'] = utils.get_central_crop(net['conv_2'], box=crop_size)

                net['concat'] = tf.concat([net['conv_2_cropped'], net['pool_2']], 3)
    return net





def model_actor(images, inits,num_iterations=3, num_patches=68, patch_shape=(30, 30)):
    bs_images, height, width, num_channels = images.get_shape().as_list()

    batch_size = tf.shape(inits)[0]
    # if bs_images != batch_size:
    #   knn_2D = tf.reshape(tf.tile(knn_2D, [batch_size, 1, 1, 1]), [batch_size, height, width, num_channels])
    print(images.get_shape().as_list())
    hidden_state = tf.zeros((batch_size, 512))
    dx = tf.zeros((batch_size, num_patches, 2))
    endpoints = {}
    dxs = []
    # state_3D = tf.zeros((batch_size, 168))
    # zero_out_module = tf.load_op_library('zero_out.so')
    # with tf.Session(''):
    #  zero_out_module.zero_out([[1, 2], [3, 4]]).eval()

    # m_module = tf.load_op_library('extract_patches_gpu.so')
    m_module = tf.load_op_library('./extract_patches.so')

    for step in range(num_iterations):
        with tf.device('/cpu:0'):
            patches = m_module.extract_patches(images, tf.constant(patch_shape, dtype=tf.int32), tf.add(inits, dx))
            tf.stop_gradient(patches)
        patches = tf.reshape(patches, (batch_size * num_patches, patch_shape[0], patch_shape[1], num_channels))
        endpoints['patches'] = patches

        print('Actor-------------Extract Patches\n')

        with tf.variable_scope('convnet', reuse=step > 0):
            net = conv_model(patches)
            feat_maps = net['concat']

        _, h, w, c = feat_maps.get_shape().as_list()  # mdzz
        # print(_, h, w, c)

        flatten_feat_maps = tf.reshape(feat_maps, (batch_size, num_patches * h * w * c))

        with tf.variable_scope('rnn', reuse=step > 0) as scope:
            hidden_state = slim.ops.fc(tf.concat([flatten_feat_maps, hidden_state], 1), 512, activation=tf.tanh)
            tf.stop_gradient(hidden_state)
            prediction = slim.ops.fc(hidden_state, num_patches * 2, scope='pred', activation=None)
            endpoints['prediction'] = prediction
        prediction = tf.reshape(prediction, (batch_size, num_patches, 2))
        dx += prediction
        dxs.append(dx)

    return dx, dxs, endpoints


def model_critic(images, shapes, patch_shape=(26, 26), num_patches=68):
    # assert k == shapes.shape[0] == 1 # k, 68, 2
    bs_images, height, width, num_channels = images.get_shape().as_list()
    batch_size = tf.shape(shapes)[0]
    state_3D = tf.zeros((batch_size, 168))
    hidden_state = tf.zeros((batch_size, 512))
    print (bs_images, batch_size)

    m_module = tf.load_op_library('./extract_patches.so')

    # pad_output = tf.reshape(pad_output, (batch_size * num_patches, height, width, num_channels))
    with tf.device('/cpu:0'):
        patches = m_module.extract_patches(images, tf.constant(patch_shape, dtype=tf.int32), shapes)
    patches = tf.reshape(patches, (batch_size * num_patches, patch_shape[0], patch_shape[1], num_channels))

    print('Critic-------------Extract Patches\n')

    with tf.variable_scope('convnet'):
        net = conv_model(patches)
        feat_maps = net['concat']
    _, h, w, c = feat_maps.get_shape().as_list()  # mdzz
    flatten_feat_maps = tf.reshape(feat_maps, (batch_size, num_patches * h * w * c))

    with tf.variable_scope('fc'):
        # fc1 = slim.ops.fc(flatten_feat_maps, 1024, activation=None)
        fc2 = slim.ops.fc(flatten_feat_maps, 512, activation=None)
        fc3 = slim.ops.fc(fc2, num_patches * 2, activation=None)
        pred = slim.ops.fc(fc3, 1, activation=None)

    return pred











