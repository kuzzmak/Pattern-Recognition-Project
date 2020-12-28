import cv2 as cv
import numpy as np

import torch

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


def resize_svoi(svoi: np.ndarray, preferred_square_size: tuple) -> np.ndarray:
    """
    Resizes all squares in one SVOI.

    Parameters
    ----------
    svoi: np.ndarray
        SVOI which squares are being resized
    preferred_square_size: tuple
        new size of squares in SVOI

    Returns
    -------
    resized_svoi: np.ndarray
        new SVOI with resized squares
    """

    assert len(preferred_square_size) == 2, "square size in SVOI must be two dimensional"
    new_shape = (svoi.shape[0], preferred_square_size[0], preferred_square_size[1])
    resized_svoi = np.zeros(new_shape, dtype='uint8')
    for dim in range(svoi.shape[0]):
        resized_svoi[dim, :, :] = resize_image(svoi[dim, :, :], preferred_square_size)
    return resized_svoi


def make_tensor_from_svoi(svoi: np.ndarray) -> torch.tensor:
    """
    Converts numpy array of shape (temporal_length, x, y) to (batch_size, temporal_length, depth, x, x),
    where batch_size if number of SVOI samples for iteration (1 here) and depth is how many colors
    does one square have (1 here, only grayscale images).
    After conversion, torch.tensor because it's convenient for input into CNN.

    Parameters
    ----------
    svoi: np.ndarray
        SVOI
    Returns
    -------
    input_tensor: torch.tensor
        tensor for input into CNN

    """
    # 5D: [batch_size, channels, depth, height, width]
    new_svoi = np.zeros((1, 1, *svoi.shape), dtype=svoi.dtype)
    for dim in range(svoi.shape[0]):
        new_svoi[0, 0, dim, :, :] = svoi[dim, :, :]
    return torch.from_numpy(new_svoi.astype(np.float32))
