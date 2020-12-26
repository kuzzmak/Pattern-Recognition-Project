import cv2 as cv
import numpy as np


def read_image(path, flag=cv.IMREAD_GRAYSCALE):
    """
    Reads image from path.

    :param path: path to the image
    :param flag: format in which image is read
    :return: image
    """
    return cv.imread(path, flag)


def new_image_shape(old_shape, square_size):
    """
    Returns new images shape where each side of the image
    is multiple of the square_size.

    :param old_shape: old shape of the image
    :param square_size: square size in image
    :return: resized old image shape
    """
    new_x = int(square_size * np.ceil(old_shape[0] / square_size))
    new_y = int(square_size * np.ceil(old_shape[1] / square_size))
    return new_x, new_y


def resize_image(image, preferred_size):
    """
    Resizes image to the new shape.

    :param image: image to be resized
    :param preferred_size: new preferred image size
    :return: resized image
    """
    return cv.resize(image, preferred_size, interpolation=cv.INTER_AREA)
