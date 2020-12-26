import cv2 as cv
import numpy as np

import os


def read_image(path: str, flag=cv.IMREAD_GRAYSCALE) -> np.ndarray:
    """
    Reads image from path.

    :param path: path to the image
    :param flag: format in which image is read
    :return: image
    """
    return cv.imread(path, flag)


def new_image_shape(old_shape: tuple, square_size: int) -> tuple:
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


def resize_image(image: np.ndarray, preferred_size: tuple) -> np.ndarray:
    """
    Resizes image to the new shape.

    :param image: image to be resized
    :param preferred_size: new preferred image size
    :return: resized image
    """
    return cv.resize(image, preferred_size, interpolation=cv.INTER_AREA)


def get_image_paths(folder_path: str, ext: str) -> list:
    """
    Function for getting all image path in folder.

    :param folder_path: path to the folder with images
    :param ext: extension of the images
    :return: list of image paths
    """
    image_names = [x for x in os.listdir(folder_path) if x.endswith(ext)]
    image_paths = [os.path.join(folder_path, x) for x in image_names]
    return image_paths
