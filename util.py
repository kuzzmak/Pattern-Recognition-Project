import cv2 as cv
import numpy as np

import os


def read_image(path: str, flag=cv.IMREAD_GRAYSCALE) -> np.ndarray:
    """
    Reads image from path.

    Parameters
    ----------
    path: str
        path to the image
    flag: int, optional
        format in which image is read

    Returns
    -------
    image: np.ndarray
        image that was read
    """

    return cv.imread(path, flag)


def new_image_shape(old_shape: tuple, square_size: int) -> tuple:
    """
    Returns new images shape where each side of the image
    is multiple of the square_size.

    Parameters
    ----------
    old_shape: tuple
        old shape of the image
    square_size: int
        square size in image

    Returns
    -------
    new_shape: tuple
        resized old image shape
    """

    new_x = int(square_size * np.ceil(old_shape[0] / square_size))
    new_y = int(square_size * np.ceil(old_shape[1] / square_size))
    return new_x, new_y


def resize_image(image: np.ndarray, preferred_size: tuple) -> np.ndarray:
    """
    Resizes image to the new shape.

    Parameters
    ----------
    image: np.ndarray
        image to be resized
    preferred_size: int
        new preferred image size

    Returns
    -------
    image: np.ndarray
        image that was resized
    """

    return cv.resize(image, preferred_size, interpolation=cv.INTER_AREA)


def get_image_paths(folder_path: str, ext: str) -> list:
    """
    Function for getting all image path in folder.

    Parameters
    ----------
    folder_path: str
        path to the folder with images
    ext: str
        extension of the images in folder

    Returns
    -------
    list: list
        list of image paths
    """

    image_names = [x for x in os.listdir(folder_path) if x.endswith(ext)]
    image_paths = [os.path.join(folder_path, x) for x in image_names]
    return image_paths
