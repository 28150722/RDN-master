from datetime import datetime
from menpofit.fitter import (noisy_shape_from_bounding_box,
                             align_shape_with_bounding_box)
import data_provider as data_provider
from menpo.shape.pointcloud import PointCloud
import joblib
import rdn_model
import numpy as np
#import skimage.util as ski
import os.path
import slim
import tensorflow as tf
import time
import utils
import menpo
import menpo.io as mio
import random

# from DDPG import Actor, Critic, Memory
import DDPG

import gc
# import objgraph
import cv2
import scipy.io as sio
import os.path
from tensorflow.python.framework import ops as op
from pathlib import Path
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# @ops.RegisterGradient("PadPatches")
# def pad_patches_grad(op, grad):
#     knn_2D = op.inputs[0]
#     _, num_patches, _ =  op.inputs[2].get_shape().as_list() # k ,num_patches, 2
#     k = tf.shape(op.inputs[2])[0]
#     print('k:', k)
#     assert len(knn_2D.get_shape().as_list()) == 4
#     filtered_y, filtered_x = utils.sobel_filter(knn_2D[0,:,:,:]) # height, width, 3
#     print('filter shape:', filtered_x.get_shape().as_list(), filtered_y.get_shape().as_list())
#     _, height, width, _ = filtered_x.get_shape().as_list()
#     filtered_y_cal = tf.reshape(tf.tile(tf.reshape(filtered_y, [-1]), [k * num_patches]), [k, num_patches, height, width, 1])
#     filtered_x_cal = tf.reshape(tf.tile(tf.reshape(filtered_x, [-1]), [k * num_patches]), [k, num_patches, height, width, 1])
#     grad_output_y = tf.expand_dims(tf.reduce_sum(filtered_y_cal * grad, [2,3,4]),2)
#     grad_output_x = tf.expand_dims(tf.reduce_sum(filtered_x_cal * grad, [2,3,4]),2)
#     return([None, None, tf.concat([grad_output_y, grad_output_x], 2)])
@op.RegisterGradient("ExtractPatchesGpu")
def extract_patches_grad(op, grad):
    m_module = tf.load_op_library('extract_patches_gpu.so')
    images = op.inputs[0]
    patch_shape = op.inputs[1]
    shapes = op.inputs[2]
    _, num_patches, _ = op.inputs[2].get_shape().as_list()  # k ,num_patches, 2
    k = tf.shape(op.inputs[2])[0]
    print('k:', k)
    assert len(images.get_shape().as_list()) == 4
    filtered_y, filtered_x = utils.sobel_filter(images)  # height, width, 3
    print('filter shape:', filtered_x.get_shape().as_list(), filtered_y.get_shape().as_list())
    # _, height, width, _ = filtered_x.get_shape().as_list()
    # filtered_y_cal = tf.reshape(tf.tile(tf.reshape(filtered_y, [-1]), [k * num_patches]), [k, num_patches, height, width, 1])
    # filtered_x_cal = tf.reshape(tf.tile(tf.reshape(filtered_x, [-1]), [k * num_patches]), [k, num_patches, height, width, 1])
    patches_y = m_module.extract_patches_gpu(filtered_y, patch_shape, shapes)
    patches_x = m_module.extract_patches_gpu(filtered_x, patch_shape, shapes)
    grad_output_y = tf.expand_dims(tf.reduce_mean(patches_y * grad, [2, 3, 4]), 2)
    grad_output_x = tf.expand_dims(tf.reduce_mean(patches_x * grad, [2, 3, 4]), 2)
    return ([None, None, tf.concat([grad_output_y, grad_output_x], 2)])



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_integer('batch_size', 30, """The batch size to use.""")
# tf.app.flags.DEFINE_integer('num_preprocess_threads', 1,  # 5.
#                             """How many preprocess threads to use.""")
tf.app.flags.DEFINE_string('train_dir', './ckpt/test/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('pretrained_checkpoint_path', '/data2/spwu_data2/wx/wx/RDN/ckpt/pred',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")


# tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '/home/hliu/gmh/RL_FA/RDN/pre_trained/',
#                            """If specified, restore this pretrained model """
#                            """before beginning any training.""")
# tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '/home/hliu/gmh/RL_FA/RDN/logs/ckpt_pre/',
#                            """If specified, restore this pretrained model """
#                            """before beginning any training.""")

tf.app.flags.DEFINE_integer('max_steps', 40000,  # 50000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('train_device', '/gpu:0', """Device to train with.""")
tf.app.flags.DEFINE_string('datasets', ':'.join(
    (
        'databases/lfpw/trainset/*.png',
        'databases/afw/*.jpg',
        'databases/helen/trainset/*.jpg',

    )),
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('patch_size', 30, 'The extracted patch size')  # 26
#####################  hyper parameters  ####################
# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999

PATCHES_2D = 68
PATCHES_3D = 84
k_nearest = 5
MAX_EPOCHES = 10
MAX_EPISODES = 1
MAX_iteration = 4
LR_A = 0.0001  # 0.0001  # learning rate for actor
LR_C = 0.00001  # 0.00001    # learning rate for critic
GAMMA = 0.9  # 0.9     # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]  # you can try different target replacement strategies
# MEMORY_CAPACITY = 3145
num_video = 8
num_fram = 2
BATCH_SIZE = num_video * num_fram
Max_step = 20000
RENDER = False
OUTPUT_GRAPH = True
# op.NotDifferentiable("ExtractPatches")

def train(scope=''):
    """Train on dataset for a number of steps."""
    # with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.

    train_dirs = FLAGS.datasets.split(':')


    _images, _shapes, _reference_shape, pca_model, shape_space, _inits= \
        data_provider.load_images(train_dirs)
    idx_list = range(len(_images))

    def get_random_sample(num, rotation_stddev=10):
        # idx = np.random.randint(low=0, high=len(_images))
        images = []
        shapes = []
        inits =[]
        shape_3D = []
        inits_3D = []
        for i in range(FLAGS.batch_size):
            rand_num = np.random.randint(0, num)
            pixel_list = []
            shape_list = []
            # print('get_random_set:', idx)
            im = menpo.image.Image(_images[rand_num].transpose(2, 0, 1), copy=False)
            lms = _shapes[rand_num]

            init = _inits[rand_num]

            im.landmarks['PTS'] = lms

            im.landmarks['inital'] = init



            if np.random.rand() < .5:
               im = utils.mirror_image(im)

            if np.random.rand() < .5:
               theta = np.random.normal(scale=rotation_stddev)
               rot = menpo.transform.rotate_ccw_about_centre(lms, theta)
               im = im.warp_to_shape(im.shape, rot)

            pixels = im.pixels.transpose(1, 2, 0).astype('float32')

            shape = im.landmarks['PTS'].lms.points.astype('float32')

            init_2 = im.landmarks['inital'].lms.points.astype('float32')

            pixel_list.append(pixels)

            pixel_list = np.array(pixel_list)

            inits.append(init_2)
            images.append(pixel_list)
            shapes.append(shape)



        return images, shapes, inits




    print('Defining model...')

    # all placeholder for tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


    with tf.device('/gpu:0'):

        actor = DDPG.Actor(sess, shape_space, k_nearest, LR_A, REPLACEMENT)

        critic = DDPG.Critic(sess, LR_C, GAMMA, REPLACEMENT)


    for var in tf.global_variables():
        print(var.op.name, var)
    print('------')
    for var in tf.trainable_variables():
        print(var.op.name, var)

    print('------')
    for var in tf.moving_average_variables():
        print(var.op.name, var)
    print('------')

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()
    print('Initializing variables...')
    sess.run(init)
    print('Initialized variables.')


    if FLAGS.pretrained_checkpoint_path:
        variables_to_restore = tf.get_collection(
            slim.variables.VARIABLES_TO_RESTORE)
        ckpt = tf.train.get_checkpoint_state(FLAGS.pretrained_checkpoint_path)
        var_name_list = ['convnet/conv_1/weights', 'convnet/conv_1/biases'
            , 'convnet/conv_2/weights', 'convnet/conv_2/biases'

            , 'rnn/FC/weights', 'rnn/FC/biases'
            , 'rnn/pred/weights', 'rnn/pred/biases'
]
        for var in variables_to_restore:
            if '/'.join(var.op.name.split('/')[2:]) in var_name_list:
                restorer = tf.train.Saver({'/'.join(var.op.name.split('/')[2:]): var})
                restorer.restore(sess, os.path.join(FLAGS.pretrained_checkpoint_path,
                                                    ckpt.model_checkpoint_path))
                print(var.op.name)
        print('%s: Pre-trained knn_2D model restored from %s' %
              (datetime.now(), FLAGS.pretrained_checkpoint_path))


    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/graph", sess.graph)
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, '')
    summary_writer = tf.summary.FileWriter("logs/")
    saver = tf.train.Saver()

    num = shape_space.shape[0]
    loss_list = []
    loss_a_list = []
    idx_save = 0
    for step in range(Max_step):
        loss = 0
        random.shuffle(idx_list)
        image_train, shape_gt, inits = get_random_sample(len(idx_list))


        for i in range(MAX_iteration):
            b_a_hat = []

            b_a_ = []

            q = []
            e = []

            states_2 = []

            for bs in range(FLAGS.batch_size):

                image = image_train[bs]
                image = np.reshape(image,(1, 386 ,458,3))  # 383, 453   386,458

                '"knn_2D initial"'


                state_2D = np.reshape(actor.choose_action_hat(inits[bs].reshape(1, PATCHES_2D, 2), image),(1, 136))


                b_hat_k_nn = np.squeeze(
                    actor.choose_action_hat(inits[bs].reshape(1, PATCHES_2D, 2), image))
                b_a_hat.append(b_hat_k_nn)

                k_nn_b_a_ = (
                    actor.choose_action(inits[bs].reshape(1, PATCHES_2D, 2), b_hat_k_nn,
                                        image))

                if random.random() < 0.5:
                    b_a_.append(np.squeeze(
                        critic.choose_max(inits[bs].reshape(1, PATCHES_2D, 2), k_nn_b_a_,
                                          image)))
                else:
                    b_a_.append(b_hat_k_nn)
                q_value = critic.q_value(inits[bs].reshape(1, PATCHES_2D, 2), b_hat_k_nn.reshape(1, PATCHES_2D, 2),
                                         image)
                error = rdn_model.normalized_rmse(
                    inits[bs].reshape(1, PATCHES_2D, 2) + b_hat_k_nn.reshape(1, PATCHES_2D, 2),
                    shape_gt[bs].reshape(1, PATCHES_2D, 2))
                states_2.append(state_2D)

                q.append(q_value)
                e.append(error)


            b_a_ = np.array(b_a_)

            b_a_hat = np.array(b_a_hat)

            images = np.array(np.squeeze(image_train))
            gts = np.array(np.squeeze(shape_gt))
            inits = np.array(np.squeeze(inits))

            e = np.vstack(e).ravel().mean()
            q = np.vstack(q).ravel().mean()


            loss = critic.learn_supervised(inits, b_a_, images, gts)

            grad_supervised, loss_a = actor.learn_supervise(inits, np.reshape(gts,
                                                                            [-1, PATCHES_2D, 2]).astype(
                'float32'), images)

            s_, r = rdn_model.step(sess, b_a_, inits, gts)
            inits = s_
            r = r.mean()

            print('Train_2D_Step:', step, 'Iterate_Steps:', i, 'R:', r, '2D_q_value:', q, '2D_error:', e, 'critic loss knn_2D:', loss, 'actor loss knn_2D :', loss_a)
        if (idx_save % 100 == 0):
            checkpoint_path = os.path.join(FLAGS.train_dir +'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=idx_save)
        idx_save += 1


if __name__ == '__main__':
    train()
