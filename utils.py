import numpy as np
from menpo.shape import PointCloud
import tensorflow as tf
from sklearn.cluster import KMeans
import cv2

jaw_indices = np.arange(0, 17)
lbrow_indices = np.arange(17, 22)
rbrow_indices = np.arange(22, 27)
upper_nose_indices = np.arange(27, 31)
lower_nose_indices = np.arange(31, 36)
leye_indices = np.arange(36, 42)
reye_indices = np.arange(42, 48)
outer_mouth_indices = np.arange(48, 60)
inner_mouth_indices = np.arange(60, 68)

jaw = np.arange(0, 33)
lbrow = np.arange(33, 38)
rbrow = np.arange(38, 43)
upper_nose = np.arange(43, 47)
lower_nose = np.arange(47, 52)
leye = np.arange(52, 58)
reye = np.arange(58, 64)
outer = np.arange(64, 76)
inner = np.arange(76, 84)

parts_68 = (jaw_indices, lbrow_indices, rbrow_indices, upper_nose_indices,
            lower_nose_indices, leye_indices, reye_indices,
            outer_mouth_indices, inner_mouth_indices)

mirrored_parts_68 = np.hstack([
    jaw_indices[::-1], rbrow_indices[::-1], lbrow_indices[::-1],
    upper_nose_indices, lower_nose_indices[::-1],
    np.roll(reye_indices[::-1], 4), np.roll(leye_indices[::-1], 4),
    np.roll(outer_mouth_indices[::-1], 7),
    np.roll(inner_mouth_indices[::-1], 5)
])

mirrored_parts_84 = np.hstack([
    jaw[::-1], rbrow[::-1], lbrow[::-1],
    upper_nose, lower_nose[::-1],
    np.roll(reye[::-1], 4), np.roll(leye[::-1], 4),
    np.roll(outer[::-1], 7),
    np.roll(inner[::-1], 5)
])


def mirror_landmarks_68(lms, image_size):
    return PointCloud(abs(np.array([0, image_size[1]]) - lms.as_vector(
    ).reshape(-1, 2))[mirrored_parts_68])
def mirror_landmarks_84(lms, image_size):
    return PointCloud(abs(np.array([0, image_size[1]]) - lms.as_vector(
    ).reshape(-1, 2))[mirrored_parts_84])

def mirror_image(im):
    im = im.copy()
    im.pixels = im.pixels[..., ::-1].copy()

    for group in im.landmarks:

        lms = im.landmarks[group].lms
        if lms.points.shape[0] == 68:
            im.landmarks[group] = mirror_landmarks_68(lms, im.shape)

        if lms.points.shape[0] == 84:
            im.landmarks[group] = mirror_landmarks_84(lms, im.shape)

    return im


def mirror_image_bb(im):
    im = im.copy()
    im.pixels = im.pixels[..., ::-1]
    im.landmarks['bounding_box'] = PointCloud(abs(np.array([0, im.shape[
        1]]) - im.landmarks['bounding_box'].lms.points))
    return im


def line(image, x0, y0, x1, y1, color):
    steep = False
    if x0 < 0 or x0 >= 400 or x1 < 0 or x1 >= 400 or y0 < 0 or y0 >= 400 or y1 < 0 or y1 >= 400:
        return

    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(int(x0), int(x1) + 1):
        t = (x - x0) / float(x1 - x0)
        y = y0 * (1 - t) + y1 * t
        if steep:
            image[x, int(y)] = color
            image[x - 1, int(y)] = color
            image[x + 1, int(y)] = color
            image[x, int(y+1)] = color
            image[x, int(y - 1)] = color
        else:
            image[int(y), x] = color
            image[int(y+1), x] = color
            image[int(y-1), x] = color
            image[int(y), x+1] = color
            image[int(y), x-1] = color

def draw_landmarks(img, lms):
    try:
        img = img.copy()
        for i, part in enumerate(parts_68[1:]):
            circular = []

            if i in (4, 5, 6, 7):
                circular = [part[0]]

            for p1, p2 in zip(part, list(part[1:]) + circular):
                p1, p2 = lms[p1], lms[p2]

                line(img, p2[1], p2[0], p1[1], p1[0], 1)#[0,1,0])
    except:
        pass
    return img

def draw_bb(img, bb):
    try:
        img = np.squeeze(img.copy())

        line(img, bb[0][1], bb[0][0], bb[1][1], bb[1][0], 1)
        line(img, bb[0][1], bb[0][0], bb[3][1], bb[3][0], 1)
        line(img, bb[2][1], bb[2][0], bb[1][1], bb[1][0], 1)
        line(img, bb[2][1], bb[2][0], bb[3][1], bb[3][0], 1)
    except:
        pass
    return img

def draw_landmarks_84(img, lms):
    try:
        img = 255 * (img.copy())
        for k in range(84):
            cv2.circle(img, (int(lms[k, 1]), int(lms[k, 0])), 3, (255, 255, 255), -1)

    except:
        pass
    return img / 255.0
def draw_landmarks_68(img, lms):
    try:
        img = 255 * (img.copy())
        for k in range(68):
            cv2.circle(img, (int(lms[k, 1]), int(lms[k, 0])), 3, (255, 255, 255), -1)

    except:
        pass
    return img / 255.0

def draw_landmarks_cv(img, lms):
    try:
        img = 255 * (img.copy())
        for i, part in enumerate(parts_68[1:]):
            circular = []

            if i in (4, 5, 6, 7):
                circular = [part[0]]

            for p1, p2 in zip(part, list(part[1:]) + circular):
                p1, p2 = lms[p1], lms[p2]
                cv2.line(img, (int(p2[1]), int(p2[0])), (int(p1[1]), int(p1[0])), (255, 255, 255), 3)

    except:
        pass
    return img / 255.0

def batch_draw_landmarks(imgs, lms):
    return np.array([draw_landmarks(img, l) for img, l in zip(imgs, lms)])

def batch_draw_landmarks_84(imgs, lms):
    return np.array([draw_landmarks_84(img, l) for img, l in zip(imgs, lms)])
def batch_draw_landmarks_68(imgs, lms):
    return np.array([draw_landmarks_68(img, l) for img, l in zip(imgs, lms)])

def batch_draw_landmarks_cv(imgs, lms):
    return np.array([draw_landmarks_cv(img, l) for img, l in zip(imgs, lms)])


def get_central_crop(images, box=(6, 6)):
    _, w, h, _ = images.get_shape().as_list()

    half_box = (box[0] / 2., box[1] / 2.)

    a = slice(int((w // 2) - half_box[0]), int((w // 2) + half_box[0]))
    b = slice(int((h // 2) - half_box[1]), int((h // 2) + half_box[1]))

    return images[:, a, b, :]


def build_sampling_grid(patch_shape):
    patch_shape = np.array(patch_shape)
    patch_half_shape = np.require(np.round(patch_shape / 2), dtype=int)
    start = -patch_half_shape
    end = patch_half_shape
    sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
    return sampling_grid.swapaxes(0, 2).swapaxes(0, 1)


default_sampling_grid = build_sampling_grid((30, 30))


def extract_patches(pixels, centres, sampling_grid=default_sampling_grid):
    """ Extracts patches from an image.

    Args:
        pixels: a numpy array of dimensions [width, height, channels]
        centres: a numpy array of dimensions [num_patches, 2]
        sampling_grid: (patch_width, patch_height, 2)

    Returns:
        a numpy array [num_patches, width, height, channels]
    """
    pixels = pixels.transpose(2, 0, 1)

    max_x = pixels.shape[-2] - 1
    max_y = pixels.shape[-1] - 1

    patch_grid = (sampling_grid[None, :, :, :] + centres[:, None, None, :]
                  ).astype('int32')

    X = patch_grid[:, :, :, 0].clip(0, max_x)
    Y = patch_grid[:, :, :, 1].clip(0, max_y)

    return pixels[:, X, Y].transpose(1, 2, 3, 0)

def sobel_filter(image):
    sobel_x = tf.constant([[-1,0,1], [-2,0,2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3,3,1,1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1,0,2,3])
    # assert len(image.get_shape().as_list()) == 3
    assert image.get_shape().as_list()[-1] == 3
    image = tf.image.rgb_to_grayscale(image)
    # image = tf.expand_dims(image, 0)
    filtered_x = tf.nn.conv2d(image, sobel_x_filter, strides=[1,1,1,1], padding='SAME')
    filtered_y = tf.nn.conv2d(image, sobel_y_filter, strides=[1,1,1,1], padding='SAME')
    return filtered_y, filtered_x

def k_means(shapes, k, num_patches=68):
    dataMat = shapes.reshape(-1, num_patches*2)
    return KMeans(n_clusters=k, random_state=0).fit(dataMat).cluster_centers_
def k_means_3D(shapes, k, num_patches=84):
    dataMat = shapes.reshape(-1, num_patches*2)
    return KMeans(n_clusters=k, random_state=0).fit(dataMat).cluster_centers_