from typing import Generator

import cv2 as cv
import numpy as np

import torch

import os
import re

UCSD = 'UCSD'
UCSD_PATH = os.path.join('data', UCSD)
UCSD_PED1_PATH = os.path.join(UCSD_PATH, 'ped1')
UCSD_PED2_PATH = os.path.join(UCSD_PATH, 'ped2')
UCSD_PED1_LABELS_PATH = os.path.join(UCSD_PED1_PATH, 'Test', 'ped1_test_labels.txt')
UCSD_PED2_LABELS_PATH = os.path.join(UCSD_PED2_PATH, 'Test', 'ped2_test_labels.txt')


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


def get_image_names_from_folder(folder, ext):
    """
    Used for getting all image names from specific folder that end
    with specific extension.

    Parameters
    ----------
    folder: str
        path to folder with images
    ext: str
        extension of images

    Returns
    -------
    image_names: list
        names of images that end with ext

    """

    image_names = [x for x in os.listdir(folder) if x.endswith(ext)]
    return image_names


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

    image_names = get_image_names_from_folder(folder_path, ext)
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


def get_labels_ucsd(dataset, ext, test_num) -> list:
    """
    Function for getting labels of one particular dataset in test folder
    of the ped1 or ped2 datasets.

    Parameters
    ----------
    dataset: str
        where is dataset located? in ped1 or ped2?
    ext: str
        extension of pictures in this dataset
    test_num: int
        for which test in test folder do we wand labels

    Returns
    -------
    labels: list
        list of ones and zeros where ones represent abnormal
        frames and zeros represent normal
    """

    test_folder = os.path.join("data", "UCSD", f"{dataset}", "Test")
    labels_file = open(os.path.join(test_folder, f"{dataset}_test_labels.txt"), 'r')
    dataset_folder = os.path.join(test_folder, "Test{:03d}".format(test_num))
    num_of_pics = len([x for x in os.listdir(dataset_folder) if x.endswith(ext)])
    num_of_datasets = len([x for x in os.listdir(test_folder) if re.match('Test[0-9]{3}$', x)])

    line_num = 1

    labels = []
    while True:
        if line_num > num_of_datasets:
            break

        line = labels_file.readline()

        if not line:
            break

        if line_num == test_num:

            labels = [0] * num_of_pics

            parts = line.split(",")
            for p in parts:
                split = p.strip().split(":")
                lower, upper = int(split[0]), int(split[1])
                labels[lower - 1: upper] = [1] * (upper - lower + 1)
            break

        line_num += 1

    labels_file.close()

    return labels


def get_labels_from_dataset(dataset_function, dataset_params):
    """
    Function which returns corresponding function for generating labels.

    Parameters
    ----------
    dataset_function
        which dataset function to use
    dataset_params: dict
        parameters for dataset function

    Returns
    -------
    dataset_function
        function which generates labels for one particular dataset
    """
    return dataset_function(**dataset_params)


def labels_generator(temporal_length, dataset_function, dataset_params) -> Generator:
    """
    Generic function which generates labels for one particular dataset.

    Parameters
    ----------
    temporal_length: int
        length in time domain od the SVOI
    dataset_function
        function which is used for extracting labels
    dataset_params: dict
        parameters needed for function which extracts labels

    Returns
    -------
    list_of_labels: list
        list of labels of each frame in SVOI
    """

    labels = get_labels_from_dataset(dataset_function, dataset_params)
    current = 0

    while True:

        if current >= len(labels):
            return

        current_labels = labels[current: current + temporal_length]
        current += temporal_length

        yield current_labels
