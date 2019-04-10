from datetime import datetime
import data_provider as data_provider
from data_provider import grey_to_rgb
# import joblib
from pathlib import Path
import rdn_model as rdn_model
import numpy as np
import os.path
import slim
import tensorflow as tf
import time
import utils
import menpo
import menpo.io as mio
import random
from tensorflow.python.framework import ops
import DDPG as DDPG


import cv2
from menpo.shape.pointcloud import PointCloud

from menpofit.fitter import (noisy_shape_from_bounding_box,
                             align_shape_with_bounding_box)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'ckpt/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/data2/spwu_data2/wx/wx/RDN/ckpt/test',
                           """Directory where to read model checkpoints.""")

tf.app.flags.DEFINE_string('dataset_path',

       "./databases/ibug"
      ,
        """The dataset path to evaluate.""")

tf.app.flags.DEFINE_string('device', '/cpu:0', 'the device to eval on.')
tf.app.flags.DEFINE_integer('patch_size', 30, 'The extracted patch size')  # 26

k_nearest = 5
REPLACEMENT = [dict(name='soft', tau=0.01), dict(name='hard', rep_iter_a=600, rep_iter_c=500)][1]
LR_A = 0
LR_C = 0
GAMMA = 0.9
MAX_EP_STEPS = 4
PATCHES_2D = 68
alpha = 0.6
def align_shapes(im, reference_shape, init=True, bb_hat=None):
    reference_shape = PointCloud(reference_shape)
    if init:
        bb = im.landmarks['bb'].lms.bounding_box()
        im.landmarks['__initial'] = align_shape_with_bounding_box(reference_shape, bb)

        im = im.rescale_to_pointcloud(reference_shape, group='__initial')
        lms = im.landmarks['PTS'].lms

        init = im.landmarks['__initial'].lms

        bb_hat = im.landmarks['bb'].lms
        # im = im.resize((235,200))
        pixels = grey_to_rgb(im).pixels.transpose(1, 2, 0).copy()
        height, width = pixels.shape[:2]

        padded_image = np.random.rand(395, 467, 3).astype(np.float32)
        dy = max(int((395 - height - 1) / 2), 0)
        dx = max(int((467 - width - 1) / 2), 0)
        pts = lms.points

        pts[:, 0] += dy
        pts[:, 1] += dx

        init_pts = init.points

        init_pts[:, 0] += dy
        init_pts[:, 1] += dx

        bb_pts = bb_hat.points
        bb_pts[:, 0] += dy
        bb_pts[:, 1] += dx

        lms = lms.from_vector(pts)
        init = init.from_vector(init_pts)

        bb_hat = bb_hat.from_vector(bb_pts)

        padded_image[dy:(height + dy), dx:(width + dx), :] = pixels
        gt = lms.points.astype(np.float32)
        init = init.points.astype(np.float32)

        return np.expand_dims(padded_image, 0), np.expand_dims(init, 0), np.expand_dims(gt, 0), bb_hat.bounding_box()
    else:
        bb = bb_hat
        # print(bb.points)
        im.landmarks['a'] = align_shape_with_bounding_box(reference_shape, bb)
        init = im.landmarks['a'].lms
        init = init.points.astype(np.float32)
        # print(PointCloud(init).bounding_box().points)
        return np.expand_dims(init, 0)


def evaluate(dataset_path):
    train_dir = Path(FLAGS.checkpoint_dir)
    reference_shape = mio.import_pickle(Path(FLAGS.checkpoint_dir) / 'reference_shape.pkl')


    print(train_dir)
    shape_space = np.load(FLAGS.checkpoint_dir + '/shape_space.npy')

    images = data_provider.load_images_test(dataset_path, reference_shape)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    with tf.device('/cpu:0'):

        actor = DDPG.Actor(sess, shape_space, k_nearest, 0, REPLACEMENT)

        critic = DDPG.Critic(sess, 0, GAMMA, REPLACEMENT, k_nearest)

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables())
    if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
            # Restores from checkpoint with absolute path.
            print('ok')
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Succesfully loaded model from %s' %
                  (ckpt.model_checkpoint_path))
        else:
            # Restores from checkpoint with relative path.
            saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                             ckpt.model_checkpoint_path))
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Succesfully loaded model from %s at step=%s.' %
                  (ckpt.model_checkpoint_path, global_step))
    else:
        print('No checkpoint file found')
        return

    errors = []
    errors_show = []

    pred_2D = np.zeros((68,2))

    for i in range(len(images)):
        print (i, '+++++++++++++++++++++++++++++++++++++++++')
        image_test = images[i]

        image_test, init, gt_shape_test, bb_hat = align_shapes(image_test, reference_shape)
        s = init

        a = np.zeros(s.shape)

        q_2D = 100


        for j in range(MAX_EP_STEPS):

            s = s + a
            a_hat = actor.choose_action_hat(s.reshape(1, PATCHES_2D, 2), image_test)
            b_hat_k_nn = np.squeeze(a_hat)
            k_nn_b_a_3_1 = (
            actor.choose_action(s.reshape(1, PATCHES_2D, 2), b_hat_k_nn,
                                       image_test))

            q = critic.q_value(s, a_hat, image_test)
            a = align_shapes(images[i], np.squeeze(s + a ), False, PointCloud(
                PointCloud(np.squeeze(s + a_hat)).bounding_box().points * alpha + bb_hat.points * (
                            1 - alpha)).bounding_box()) - s
            error = rdn_model.normalized_rmse(s+ a_hat, gt_shape_test)
            print('===========', q[0][0], error)
            if q <= q_2D:
                 q_2D = q
                 pred_2D = s + a_hat

        pred = pred_2D
        error = rdn_model.normalized_rmse(pred, gt_shape_test)
        print (error)
        errors.append(error)
        errors_nn = np.vstack(errors).ravel()
    for i, e in enumerate(errors_show):
        print(i, e)
    #errors = np.vstack(errors).ravel()
    errors_ = np.vstack(errors).ravel()
    print(errors_)
    mean_rmse = errors_.mean()
    auc_at_08 = (errors_ < .08).mean()
    auc_at_05 = (errors_ < .05).mean()
    print('mean_rmse = %.4f, auc @ 0.05 = %.4f, auc @ 0.08 = %.4f' %
          (mean_rmse, auc_at_05, auc_at_08))
if __name__ == '__main__':
    evaluate(FLAGS.dataset_path.split(':'))
