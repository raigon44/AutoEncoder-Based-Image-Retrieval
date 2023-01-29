import cv2
import matplotlib.pyplot as plt
from Datapreperation import prepare_data
import tensorflow as tf
import numpy as np


def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):

    mode = mode.lower()
    if image.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
    if seed is not None:
        np.random.seed(seed=seed)

    if mode == 'gaussian':
        noise = np.random.normal(kwargs['mean'], kwargs['var'] ** 0.5,
                                 image.shape)
        out = image + noise
    if clip:
        out = np.clip(out, low_clip, 1.0)

    return out


def projective_transformation(image):

    src_points = np.float32([[0, 0], [32 - 1, 0], [0, 32 - 1], [32 - 1, 32 - 1]])
    dst_points = np.float32([[0, 0], [32 - 1, 0], [int(0.33 * 32), 32 - 1], [int(0.66 * 32), 32 - 1]])

    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    img_output = cv2.warpPerspective(image, projective_matrix, (32, 32))

    return img_output


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = prepare_data()

    original_image = x_train[np.random.choice(50000, size=1, replace=False)].reshape(32, 32, 3)

    ax = plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 2, 2)
    plt.imshow(random_noise(original_image, 'gaussian', mean=0.1, var=0.01).reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 2, 3)
    plt.imshow(original_image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 2, 4)
    plt.imshow(projective_transformation(original_image).reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.show()




