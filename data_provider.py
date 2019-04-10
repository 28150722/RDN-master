from functools import partial
from menpo.shape.pointcloud import PointCloud
from menpofit.builder import compute_reference_shape
from menpofit.builder import rescale_images_to_reference_shape
from menpofit.fitter import (noisy_shape_from_bounding_box,
                             align_shape_with_bounding_box)
from pathlib import Path
import random
import os
import joblib
import menpo.feature
import menpo.image
import menpo.io as mio
import numpy as np
import tensorflow as tf
import detect
import utils
from matplotlib import pyplot as plt
import cv2
import pdb
import scipy.io as sio
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
data_list = []


def build_reference_shape(paths, diagonal=200):
    """Builds the reference shape.

    Args:
      paths: paths that contain the ground truth landmark files.
      diagonal: the diagonal of the reference shape in pixels.
    Returns:
      the reference shape.
    """
    landmarks = []
    for path in paths:
        path = Path(path).parent.as_posix()
        landmarks += [
            group.lms
            for group in mio.import_landmark_files(path, verbose=True)
            if group.lms.n_points == 68
        ]

    return compute_reference_shape(landmarks,
                                   diagonal=diagonal).points.astype(np.float32)
def build_reference_3D(paths, diagonal=200):
    """Builds the reference shape.

    Args:
      paths: paths that contain the ground truth landmark files.
      diagonal: the diagonal of the reference shape in pixels.
    Returns:
      the reference shape.
    """
    landmarks = []
    for path in paths:
        path = Path(path).parent.as_posix()
        landmarks += [
            group.lms
            for group in mio.import_landmark_files(path, verbose=True)
            if group.lms.n_points == 84
        ]

    return compute_reference_shape(landmarks,
                                   diagonal=diagonal).points.astype(np.float32)

def grey_to_rgb(im):
    """Converts menpo Image to rgb if greyscale

    Args:
      im: menpo Image with 1 or 3 channels.
    Returns:
      Converted menpo `Image'.
    """
    assert im.n_channels in [1, 3]

    if im.n_channels == 3:
        return im

    im.pixels = np.vstack([im.pixels] * 3)
    return im


def align_reference_shape(reference_shape, bb):
    min_xy = tf.reduce_min(reference_shape, 0)
    max_xy = tf.reduce_max(reference_shape, 0)
    min_x, min_y = min_xy[0], min_xy[1]
    max_x, max_y = max_xy[0], max_xy[1]

    reference_shape_bb = tf.stack([[min_x, min_y], [max_x, min_y],
                                   [max_x, max_y], [min_x, max_y]])

    def norm(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x - tf.reduce_mean(x, 0))))

    ratio = norm(bb) / norm(reference_shape_bb)
    return tf.add(
        (reference_shape - tf.reduce_mean(reference_shape_bb, 0)) * ratio,
        tf.reduce_mean(bb, 0),
        name='initial_shape')


def random_shape(gts, reference_shape, pca_model):
    """Generates a new shape estimate given the ground truth shape.

    Args:
      gts: a numpy array [num_landmarks, 2]
      reference_shape: a Tensor of dimensions [num_landmarks, 2]
      pca_model: A PCAModel that generates shapes.
    Returns:
      The aligned shape, as a Tensor [num_landmarks, 2].
    """

    def synthesize(lms):
        return detect.synthesize_detection(pca_model, menpo.shape.PointCloud(
            lms).bounding_box()).points.astype(np.float32)

    bb, = tf.py_func(synthesize, [gts], [tf.float32])
    shape = align_reference_shape(reference_shape, bb)
    shape.set_shape(reference_shape.get_shape())

    return shape


def get_noisy_init_from_bb(reference_shape, bb, noise_percentage=.02):
    """Roughly aligns a reference shape to a bounding box.

    This adds some uniform noise for translation and scale to the
    aligned shape.

    Args:
      reference_shape: a numpy array [num_landmarks, 2]
      bb: bounding box, a numpy array [4, ]
      noise_percentage: noise presentation to add.
    Returns:
      The aligned shape, as a numpy array [num_landmarks, 2]
    """
    bb = PointCloud(bb)
    reference_shape = PointCloud(reference_shape)

    bb = noisy_shape_from_bounding_box(
        reference_shape,
        bb,
        noise_percentage=[noise_percentage, 0, noise_percentage]).bounding_box(
    )

    return align_shape_with_bounding_box(reference_shape, bb).points


def load_images(paths, group1=None,group2=None, verbose=True, PLOT=False, AFLW=False, PLOT_shape=True):
    """Loads and rescales input knn_2D to the diagonal of the reference shape.

    Args:
      paths: a list of strings containing the data directories.
      reference_shape (meanshape): a numpy array [num_landmarks, 2]
      group: landmark group containing the grounth truth landmarks.
      verbose: boolean, print debugging info.
    Returns:
      knn_2D: a list of numpy arrays containing knn_2D.
      shapes: a list of the ground truth landmarks.
      reference_shape (meanshape): a numpy array [num_landmarks, 2].
      shape_gen: PCAModel, a shape generator.
    """
    images = []
    shapes = []

    bbs = []
    inits = []

    shape_space = []
    plot_shape_x = []
    plot_shape_y = []
    # compute mean shape
    # if AFLW:
    #     # reference_shape = PointCloud(mio.import_pickle(Path('/home/hliu/gmh/RL_FA/mdm_aflw/ckpt/train_aflw') / 'reference_shape.pkl'))
    #     reference_shape = mio.import_pickle(
    #         Path('/home/hliu/data2/CongcongZhu/ICME/RDN/ckpt/pred') / 'reference_shape.pkl')
    # else:
    reference_shape = PointCloud(build_reference_shape(paths))


    for path in paths:
        if verbose:
            print('Importing data from {}'.format(path))

        for im in mio.import_images(path, verbose=verbose, as_generator=True):
            # group = group or im.landmarks[group]._group_label
            # group = group or im.landmarks.keys()[0]
            group1 = 'PTS'

            bb_root = im.path.parent.relative_to(im.path.parent.parent.parent)
            if 'set' not in str(bb_root):
                bb_root = im.path.parent.relative_to(im.path.parent.parent)

            if AFLW:
                im.landmarks['bb'] = im.landmarks['PTS'].lms.bounding_box()
            else:
                im.landmarks['bb'] = mio.import_landmark_file(str(Path(
                    'bbs') / bb_root / (im.path.stem + '.pts')))
            im = im.crop_to_landmarks_proportion(0.3, group='bb')
            bb = im.landmarks['bb'].lms.bounding_box()

            im.landmarks['initial'] = align_shape_with_bounding_box(reference_shape,
                                                                    bb)

            im = im.rescale_to_pointcloud(reference_shape, group=group1)
            im = grey_to_rgb(im)
            images.append(im.pixels.transpose(1, 2, 0))
            inits.append(im.landmarks['initial'].lms)

            shapes.append(im.landmarks[group1].lms)

            shape_space.append(im.landmarks[group1].lms.points)
            bbs.append(im.landmarks['bb'].lms)
            if PLOT_shape:
                x_tmp = np.sum((im.landmarks[group1].lms.points[:, 0] - reference_shape.points[:, 0]))
                y_tmp = np.sum((im.landmarks[group1].lms.points[:, 1] - reference_shape.points[:, 1]))
                if x_tmp < 0 and y_tmp < 0:
                    plot_shape_x.append(x_tmp)
                    plot_shape_y.append(y_tmp)

    shape_space = np.array(shape_space)
    print('shape_space:', shape_space.shape)

    train_dir = Path(FLAGS.train_dir)

    # centers = utils.k_means(shape_space, 100)
    # centers = np.reshape(centers, [-1, 68, 2])

    # np.save(train_dir/'shape_space_origin.npy', centers)
    # print('created shape_space.npy using the {} group'.format(group))
    # exit(0)
    if PLOT_shape:
        k_nn_plot_x = []
        k_nn_plot_y = []
        centers = utils.k_means(shape_space, 100)
        centers = np.reshape(centers, [-1, 68, 2])
        for i in range(centers.shape[0]):
            x_tmp = np.sum((centers[i, :, 0] - reference_shape.points[:, 0]))
            y_tmp = np.sum((centers[i, :, 1] - reference_shape.points[:, 1]))
            if x_tmp < 0 and y_tmp < 0:
                k_nn_plot_x.append(x_tmp)
                k_nn_plot_y.append(y_tmp)
        # pdb.set_trace()
        # plt.scatter(plot_shape_x, plot_shape_y, s=20)
        # plt.scatter(k_nn_plot_x, k_nn_plot_y, s=40)
        # plt.xticks(())
        # plt.yticks(())
        # plt.show()
        # pdb.set_trace()

    mio.export_pickle(reference_shape.points, train_dir / 'reference_shape.pkl', overwrite=True)
    print('created reference_shape.pkl using the {} group'.format(group1))

    pca_model = detect.create_generator(shapes, bbs)

    # Pad knn_2D to max length
    max_shape = np.max([im.shape for im in images], axis=0)
    max_shape = [len(images)] + list(max_shape)
    padded_images = np.random.rand(*max_shape).astype(np.float32)
    print(padded_images.shape,'====================================================')

    if PLOT:
        # plot without padding
        centers = utils.k_means(shape_space, 100)
        centers = np.reshape(centers, [-1, 68, 2])
        plot_img = cv2.imread('a.png').transpose(2, 0, 1)
        centers_tmp = np.zeros(centers.shape)
        # menpo_img = mio.import_image('a.png')
        menpo_img = menpo.image.Image(plot_img)
        for i in range(centers.shape[0]):
            menpo_img.view()
            min_y = np.min(centers[i, :, 0])
            min_x = np.min(centers[i, :, 1])
            centers_tmp[i, :, 0] = centers[i, :, 0] - min_y + 20
            centers_tmp[i, :, 1] = centers[i, :, 1] - min_x + 20
            print(centers_tmp[i, :, :])
            menpo_img.landmarks['center'] = PointCloud(centers_tmp[i, :, :])
            menpo_img.view_landmarks(group='center', marker_face_colour='b', marker_size='16')
            # menpo_img.landmarks['center'].view(render_legend=True)
            plt.savefig('plot_shape_space/' + str(i) + '.png')
            plt.close()
        exit(0)

    # !!!shape_space without delta, which means shape_space has already been padded!

    # delta = np.zeros(shape_space.shape)
    gts = []

    inis = []

    for i, im in enumerate(images):
        height, width = im.shape[:2]
        dy = max(int((max_shape[1] - height - 1) / 2), 0)
        dx = max(int((max_shape[2] - width - 1) / 2), 0)
        lms = shapes[i]
        pts = lms.points


        init = inits[i]
        init_shape = init.points


        pts[:, 0] += dy
        pts[:, 1] += dx

        init_shape[:, 0] += dy
        init_shape[:, 1] += dx

        shape_space[i, :, 0] += dy
        shape_space[i, :, 1] += dx
        # delta[i][:, 0] = dy
        # delta[i][:, 1] = dx
        lms = lms.from_vector(pts)

        init = init.from_vector(init_shape)

        padded_images[i, dy:(height + dy), dx:(width + dx)] = im

        gts.append(lms)

        inis.append(init)


    # shape_space = np.concatenate((shape_space, delta), 2)

    centers = utils.k_means(shape_space, 100)
    centers = np.reshape(centers, [-1, 68, 2])

    np.save(train_dir / 'shape_space.npy', centers)
    print('created shape_space.npy using the {} group'.format(group1))

    return padded_images, gts, reference_shape.points.astype('float32'), pca_model, centers, inis


def load_images_aflw(paths, group=None, verbose=True, PLOT=True, AFLW=False, PLOT_shape=False):
    """Loads and rescales input knn_2D to the diagonal of the reference shape.

    Args:
      paths: a list of strings containing the data directories.
      reference_shape (meanshape): a numpy array [num_landmarks, 2]
      group: landmark group containing the grounth truth landmarks.
      verbose: boolean, print debugging info.
    Returns:
      knn_2D: a list of numpy arrays containing knn_2D.
      shapes: a list of the ground truth landmarks.
      reference_shape (meanshape): a numpy array [num_landmarks, 2].
      shape_gen: PCAModel, a shape generator.
    """
    images = []
    shapes = []
    bbs = []
    shape_space = []
    plot_shape_x = []
    plot_shape_y = []
    # compute mean shape
    if AFLW:
        # reference_shape = PointCloud(mio.import_pickle(Path('/home/hliu/gmh/RL_FA/mdm_aflw/ckpt/train_aflw') / 'reference_shape.pkl'))
        reference_shape = mio.import_pickle(
            Path('/home/hliu/gmh/RL_FA/mdm_aflw/ckpt/train_aflw') / 'reference_shape.pkl')
    else:
        reference_shape = PointCloud(build_reference_shape(paths))

    for path in paths:
        if verbose:
            print('Importing data from {}'.format(path))

        for im in mio.import_images(path, verbose=verbose, as_generator=True):
            # group = group or im.landmarks[group]._group_label
            group = group or im.landmarks.keys()[0]
            bb_root = im.path.parent.relative_to(im.path.parent.parent.parent)
            if 'set' not in str(bb_root):
                bb_root = im.path.parent.relative_to(im.path.parent.parent)

            if AFLW:
                im.landmarks['bb'] = im.landmarks['PTS'].lms.bounding_box()
            else:
                im.landmarks['bb'] = mio.import_landmark_file(str(Path(
                    'bbs') / bb_root / (im.path.stem + '.pts')))
            im = im.crop_to_landmarks_proportion(0.3, group='bb')
            im = im.rescale_to_pointcloud(reference_shape, group=group)
            im = grey_to_rgb(im)
            # knn_2D.append(im.pixels.transpose(1, 2, 0))
            shapes.append(im.landmarks[group].lms)
            shape_space.append(im.landmarks[group].lms.points)
            bbs.append(im.landmarks['bb'].lms)
            if PLOT_shape:
                x_tmp = np.sum((im.landmarks[group].lms.points[:, 0] - reference_shape.points[:, 0]))
                y_tmp = np.sum((im.landmarks[group].lms.points[:, 1] - reference_shape.points[:, 1]))
                if x_tmp < 0 and y_tmp < 0:
                    plot_shape_x.append(x_tmp)
                    plot_shape_y.append(y_tmp)
    shape_space = np.array(shape_space)
    print('shape_space:', shape_space.shape)

    train_dir = Path(FLAGS.train_dir)
    if PLOT_shape:
        k_nn_plot_x = []
        k_nn_plot_y = []
        centers = utils.k_means(shape_space, 500, num_patches=19)
        centers = np.reshape(centers, [-1, 19, 2])
        for i in range(centers.shape[0]):
            x_tmp = np.sum((centers[i, :, 0] - reference_shape.points[:, 0]))
            y_tmp = np.sum((centers[i, :, 1] - reference_shape.points[:, 1]))
            if x_tmp < 0 and y_tmp < 0:
                k_nn_plot_x.append(x_tmp)
                k_nn_plot_y.append(y_tmp)

        # plt.scatter(plot_shape_x, plot_shape_y, s=20)
        # plt.scatter(k_nn_plot_x, k_nn_plot_y, s=40)
        # plt.xticks(())
        # plt.yticks(())
        # plt.show()
        # pdb.set_trace()

    np.save(train_dir / 'shape_space_all.npy', shape_space)
    # centers = utils.k_means(shape_space, 100)
    # centers = np.reshape(centers, [-1, 68, 2])

    # np.save(train_dir/'shape_space_origin.npy', centers)
    # print('created shape_space.npy using the {} group'.format(group))
    # exit(0)

    mio.export_pickle(reference_shape.points, train_dir / 'reference_shape.pkl', overwrite=True)
    print('created reference_shape.pkl using the {} group'.format(group))

    pca_model = detect.create_generator(shapes, bbs)

    # Pad knn_2D to max length
    max_shape = [272, 261, 3]
    padded_images = np.random.rand(*max_shape).astype(np.float32)
    print(padded_images.shape)

    if PLOT:
        # plot without padding
        centers = utils.k_means(shape_space, 500, num_patches=19)
        centers = np.reshape(centers, [-1, 19, 2])
        plot_img = cv2.imread('a.png').transpose(2, 0, 1)
        centers_tmp = np.zeros(centers.shape)
        # menpo_img = mio.import_image('a.png')
        menpo_img = menpo.image.Image(plot_img)
        for i in range(centers.shape[0]):
            menpo_img.view()
            min_y = np.min(centers[i, :, 0])
            min_x = np.min(centers[i, :, 1])
            centers_tmp[i, :, 0] = centers[i, :, 0] - min_y + 20
            centers_tmp[i, :, 1] = centers[i, :, 1] - min_x + 20
            print(centers_tmp[i, :, :])
            menpo_img.landmarks['center'] = PointCloud(centers_tmp[i, :, :])
            menpo_img.view_landmarks(group='center', marker_face_colour='b', marker_size='16')
            # menpo_img.landmarks['center'].view(render_legend=True)
            plt.savefig('plot_shape_space_aflw/' + str(i) + '.png')
            plt.close()
        exit(0)

    # !!!shape_space without delta, which means shape_space has already been padded!

    # delta = np.zeros(shape_space.shape)

    for i, im in enumerate(images):
        height, width = im.shape[:2]
        dy = max(int((max_shape[0] - height - 1) / 2), 0)
        dx = max(int((max_shape[1] - width - 1) / 2), 0)
        lms = shapes[i]
        pts = lms.points
        pts[:, 0] += dy
        pts[:, 1] += dx
        shape_space[i, :, 0] += dy
        shape_space[i, :, 1] += dx
        # delta[i][:, 0] = dy
        # delta[i][:, 1] = dx
        lms = lms.from_vector(pts)
        padded_images[i, dy:(height + dy), dx:(width + dx)] = im

    # shape_space = np.concatenate((shape_space, delta), 2)

    centers = utils.k_means(shape_space, 1000, num_patches=19)
    centers = np.reshape(centers, [-1, 19, 2])

    # pdb.set_trace()
    np.save(train_dir / 'shape_space.npy', centers)
    print('created shape_space.npy using the {} group'.format(group))
    exit(0)
    return padded_images, shapes, reference_shape.points, pca_model, centers


def load_images_test(paths, reference_shape, group=None, verbose=True, PLOT=False):
    """Loads and rescales input knn_2D to the diagonal of the reference shape.

    Args:
      paths: a list of strings containing the data directories.
      reference_shape (meanshape): a numpy array [num_landmarks, 2]
      group: landmark group containing the grounth truth landmarks.
      verbose: boolean, print debugging info.
    Returns:
      knn_2D: a list of numpy arrays containing knn_2D.
      shapes: a list of the ground truth landmarks.
      reference_shape (meanshape): a numpy array [num_landmarks, 2].
      shape_gen: PCAModel, a shape generator.
    """
    images = []
    shapes = []
    scales = []
    # compute mean shape
    reference_shape = PointCloud(reference_shape)
    nameList = []
    bbox = []
    data = dict()
    for path in paths:
        if verbose:
            print('Importing data from {}'.format(path))

        for im in mio.import_images(path, verbose=verbose, as_generator=True):
            # group = group or im.landmarks[group]._group_label
            group = group or im.landmarks.keys()[0]
            bb_root = im.path.parent.relative_to(im.path.parent.parent.parent)
            if 'set' not in str(bb_root):
                bb_root = im.path.parent.relative_to(im.path.parent.parent)
            im.landmarks['bb'] = mio.import_landmark_file(str(Path(
                'bbs') / bb_root / (im.path.stem.replace(' ', '') + '.pts')))

            nameList.append(str(im.path))
            lms = im.landmarks['bb'].lms.points
            bbox.append([lms[0, 1], lms[2, 1], lms[0, 0], lms[1, 0]])
            # bbox = np.array(bbox)
            # data['nameList'] = nameList
            # data['bbox'] = bbox
            # sio.savemat('ibug_data.mat', {'nameList':data['nameList'], 'bbox':data['bbox']})
            # exit(0)

            im = im.crop_to_landmarks_proportion(0.3, group='bb')
            images.append(im)

    return images


def load_images_test_300VW(paths, reference_shape, group=None, verbose=True, PLOT=False):
    """Loads and rescales input knn_2D to the diagonal of the reference shape.

    Args:
      paths: a list of strings containing the data directories.
      reference_shape (meanshape): a numpy array [num_landmarks, 2]
      group: landmark group containing the grounth truth landmarks.
      verbose: boolean, print debugging info.
    Returns:
      knn_2D: a list of numpy arrays containing knn_2D.
      shapes: a list of the ground truth landmarks.
      reference_shape (meanshape): a numpy array [num_landmarks, 2].
      shape_gen: PCAModel, a shape generator.
    """
    images = []
    shapes = []
    scales = []
    # compute mean shape
    reference_shape = PointCloud(reference_shape)

    for path in paths:
        if verbose:
            print('Importing data from {}'.format(path))

        for im in mio.import_images(path, verbose=verbose, as_generator=True):
            # group = group or im.landmarks[group]._group_label
            # pdb.set_trace()

            # bb_root = im.path.parent.relative_to(im.path.parent.parent.parent)
            bb_root = im.path.parent
            if 'set' not in str(bb_root):
                bb_root = im.path.parent.relative_to(im.path.parent.parent)
            im.landmarks['bb'] = mio.import_landmark_file(bb_root / str(Path(
                'bbs') / (im.path.stem + '.pts')))
            im.landmarks['PTS'] = mio.import_landmark_file(bb_root / str(Path(
                'annot') / (im.path.stem + '.pts')))

            im = im.crop_to_landmarks_proportion(0.3, group='bb')
            # im = im.rescale_to_pointcloud(reference_shape, group=group)
            # _, height, width = im.pixels.shape

            # im = im.resize([386, 458])
            # im = grey_to_rgb(im)
            # knn_2D.append(im.pixels.transpose(1, 2, 0))
            # shapes.append(im.landmarks[group].lms.points.astype('float32'))
            # scales.append([386/height, 485/width])
            # lms = im.landmarks[group].lms
            # im = im.pixels.transpose(1, 2, 0)
            # height, width = im.shape[:2]
            # # print('shape:', height, width)
            # padded_image = np.random.rand(386, 458, 3).astype(np.float32)
            # dy = max(int((386 - height - 1) / 2), 0)
            # dx = max(int((458 - width - 1) / 2), 0)
            # pts = lms.points
            # pts[:, 0] += dy
            # pts[:, 1] += dx
            # # delta[i][:, 0] = dy
            # # delta[i][:, 1] = dx
            # lms = lms.from_vector(pts)
            # padded_image[dy:(height+dy), dx:(width+dx), :] = im
            images.append(im)
            # shapes.append(lms.points.astype('float32'))

    return images


def load_image(path, reference_shape, is_training=False, group='PTS',
               mirror_image=False):
    """Load an annotated image.

    In the directory of the provided image file, there
    should exist a landmark file (.pts) with the same
    basename as the image file.

    Args:
      path: a path containing an image file.
      reference_shape: a numpy array [num_landmarks, 2]
      is_training: whether in training mode or not.
      group: landmark group containing the grounth truth landmarks.
      mirror_image: flips horizontally the image's pixels and landmarks.
    Returns:
      pixels: a numpy array [width, height, 3].
      estimate: an initial estimate a numpy array [68, 2].
      gt_truth: the ground truth landmarks, a numpy array [68, 2].
    """
    im = mio.import_image(path)
    bb_root = im.path.parent.relative_to(im.path.parent.parent.parent)
    if 'set' not in str(bb_root):
        bb_root = im.path.parent.relative_to(im.path.parent.parent)

    im.landmarks['bb'] = mio.import_landmark_file(str(Path('bbs') / bb_root / (
            im.path.stem + '.pts')))
    im = im.crop_to_landmarks_proportion(0.3, group='bb')
    reference_shape = PointCloud(reference_shape)

    bb = im.landmarks['bb'].lms.bounding_box()

    im.landmarks['__initial'] = align_shape_with_bounding_box(reference_shape,
                                                              bb)
    im = im.rescale_to_pointcloud(reference_shape, group='__initial')

    if mirror_image:
        im = utils.mirror_image(im)

    lms = im.landmarks[group].lms
    initial = im.landmarks['__initial'].lms

    # if the image is greyscale then convert to rgb.
    pixels = grey_to_rgb(im).pixels.transpose(1, 2, 0)

    gt_truth = lms.points.astype(np.float32)
    estimate = initial.points.astype(np.float32)
    return pixels.astype(np.float32).copy(), gt_truth, estimate


def distort_color(image, thread_id=0, stddev=0.1, scope=None):
    """Distort the color of the image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
      image: Tensor containing single image.
      thread_id: preprocessing thread ID.
      scope: Optional scope for op_scope.
    Returns:
      color-distorted image
    """
    with tf.op_scope([image], scope, 'distort_color'):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        image += tf.random_normal(
            tf.shape(image),
            stddev=stddev,
            dtype=tf.float32,
            seed=42,
            name='add_gaussian_noise')
        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def load_image_test(path, reference_shape, frame_num):
    file_name = path[:-1] + "/%06d.jpg" % (frame_num)

    im = mio.import_image(file_name)

    im.landmarks['PTS'] = mio.import_landmark_file(path[:-1] + "/annot/%06d.pts" % (frame_num))
    # im.landmarks['PTS'] = mio.import_landmark_file(path[:-1] + "/%06d.pts" % (frame_num))
    bb_path = path[:-1] + "/bbs/%06d.pts" % (frame_num)

    im.landmarks['bb'] = mio.import_landmark_file(bb_path)

    im = im.crop_to_landmarks_proportion(0.3, group='bb')
    reference_shape = PointCloud(reference_shape)

    bb = im.landmarks['bb'].lms.bounding_box()

    im.landmarks['__initial'] = align_shape_with_bounding_box(reference_shape,
                                                              bb)
    im = im.rescale_to_pointcloud(reference_shape, group='__initial')

    lms = im.landmarks['PTS'].lms
    initial = im.landmarks['__initial'].lms

    # if the image is greyscale then convert to rgb.
    pixels = grey_to_rgb(im).pixels.transpose(1, 2, 0)

    gt_truth = lms.points.astype(np.float32)
    estimate = initial.points.astype(np.float32)

    return 1, pixels.astype(np.float32).copy(), gt_truth, estimate


def load_file_list(dataset):
    file = open(dataset)
    num = 0
    for line in file:
        if num == 0:
            path = line
            num = 1
        else:
            data_list.append((path, int(line)))
            num = 0


def get_random_list(batch_size, reference_shape, num_load=1):
    images = []
    gths = []
    inits = []
    data_size = len(data_list)
    for j in range(num_load):
        data_num = random.randint(0, data_size - 1)
        data_path = data_list[data_num][0]
        # frame_stride = random.randint(1, 3)
        frame_stride = 1
        data_frame_size = data_list[data_num][1]
        frame_num = random.randint(1 - frame_stride, data_frame_size - 1 - frame_stride * (batch_size))
        for i in range(batch_size):
            frame_num = frame_num + frame_stride
            file_name = data_path[:-1] + "/%06d.jpg" % (frame_num)
            while os.path.exists(file_name) == False:
                frame_num = frame_num + 1
                file_name = data_path[:-1] + "/%06d.jpg" % (frame_num)
            get_face, image, gt_truth, estimate = load_image(data_path, reference_shape, frame_num)
            if get_face == 0:
                return get_random_list(batch_size, reference_shape)
            images.append(image)
            gths.append(gt_truth)
            inits.append(estimate)

    return images, gths, inits


def batch_inputs(paths,
                 reference_shape,
                 batch_size=1,
                 is_training=False,
                 num_landmarks=68,
                 mirror_image=False):
    """Reads the files off the disk and produces batches.

    Args:
      paths: a list of directories that contain training knn_2D and
        the corresponding landmark files.
      reference_shape: a numpy array [num_landmarks, 2]
      batch_size: the batch size.
      is_traininig: whether in training mode.
      num_landmarks: the number of landmarks in the training knn_2D.
      mirror_image: mirrors the image and landmarks horizontally.
    Returns:
      knn_2D: a tf tensor of image [batch_size, width, height, 3].
      lms: a tf tensor of shape [batch_size, 68, 2].
      lms_init: a tf tensor of shape [batch_size, 68, 2].
    """

    files = tf.concat([map(str, sorted(Path(d).parent.glob(Path(d).name)))
                       for d in paths], 0)

    filename_queue = tf.train.string_input_producer(files,
                                                    shuffle=False,
                                                    capacity=3500)

    filename = filename_queue.dequeue()

    image, lms, lms_init = tf.py_func(
        partial(load_image, is_training=is_training,
                mirror_image=mirror_image),
        [filename, reference_shape],  # input arguments
        [tf.float32, tf.float32, tf.float32],  # output types
        name='load_image'
    )

    # The image has always 3 channels.
    image.set_shape([None, None, 3])

    if is_training:
        image = distort_color(image)

    lms = tf.reshape(lms, [num_landmarks, 2])
    lms_init = tf.reshape(lms_init, [num_landmarks, 2])

    images, lms, inits, shapes = tf.train.batch(
        [image, lms, lms_init, tf.shape(image)],
        batch_size=batch_size,
        num_threads=1,
        capacity=1000,
        enqueue_many=False,
        dynamic_pad=True)

    return images, lms, inits, shapes
