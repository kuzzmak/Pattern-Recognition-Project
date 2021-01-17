from typing import Generator, List
import random
import json
import uuid
import os
import re

import cv2 as cv
import numpy as np

import torch

UCSD = 'UCSD'
PED1 = 'PED1'
PED2 = 'PED2'
UCSD_PATH = os.path.join('data', UCSD)
UCSD_PED1_PATH = os.path.join(UCSD_PATH, 'PED1')
UCSD_PED2_PATH = os.path.join(UCSD_PATH, 'PED2')
UCSD_PED1_LABELS_PATH = os.path.join(UCSD_PED1_PATH, 'Test', 'ped1_test_labels.txt')
UCSD_PED2_LABELS_PATH = os.path.join(UCSD_PED2_PATH, 'Test', 'ped2_test_labels.txt')

UMN = 'UMN'
INDOOR = 'INDOOR'
LAWN = 'LAWN'
PLAZA = 'PLAZA'
UMN_PATH = os.path.join('data', UMN)
UMN_INDOOR_PATH = os.path.join(UMN_PATH, 'indoor')
UMN_LAWN_PATH = os.path.join(UMN_PATH, 'lawn')
UMN_PLAZA_PATH = os.path.join(UMN_PATH, 'plaza')
UMN_INDOOR_LABELS_PATH = os.path.join(UMN_INDOOR_PATH, 'umn_indoor_labels.txt')
UMN_LAWN_LABELS_PATH = os.path.join(UMN_LAWN_PATH, 'umn_lawn_labels.txt')
UMN_PLAZA_LABELS_PATH = os.path.join(UMN_PLAZA_PATH, 'umn_plaza_labels.txt')

UCSD_EXT = '.tif'
UMN_EXT = '.png'

UCSD_GT_EXT = '.bmp'

DATASETS = dict(
    UCSD=dict(
        PED1=dict(
            DATASET_PATH=UCSD_PED1_PATH,
            LABELS_PATH=UCSD_PED1_LABELS_PATH,
        ),
        PED2=dict(
            DATASET_PATH=UCSD_PED2_PATH,
            LABELS_PATH=UCSD_PED2_LABELS_PATH,
        ),
    ),
    UMN=dict(
        INDOOR=dict(
            DATASET_PATH=UMN_INDOOR_PATH,
            LABELS_PATH=UMN_INDOOR_LABELS_PATH,
        ),
        LAWN=dict(
            DATASET_PATH=UMN_LAWN_PATH,
            LABELS_PATH=UMN_LAWN_LABELS_PATH,
        ),
        PLAZA=dict(
            DATASET_PATH=UMN_PLAZA_PATH,
            LABELS_PATH=UMN_PLAZA_LABELS_PATH,
        ),
    )
)

MODELS_PATH = 'models'
MODELS_DATA_PATH = os.path.join(MODELS_PATH, 'models_data.json')

ACC_FILE_PATH = os.path.join(MODELS_PATH, 'acc_file.txt')


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

    size = (preferred_size[1], preferred_size[0])
    return cv.resize(image, size, interpolation=cv.INTER_AREA)


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
    assert os.path.exists(folder), "folder does not exists"
    image_names = [x for x in os.listdir(folder) if x.endswith(ext)]
    return image_names


def get_image_paths(folder_path: str, ext: str) -> List[str]:
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


def get_labels_ucsd(dataset_params: dict) -> list:
    """
    Function for getting labels of one particular dataset in test folder
    of the PED1 or PED2 datasets.

    Parameters
    ----------
    dataset_params: dict
        dictionary of dataset parameters

    Returns
    -------
    labels: list
        list of ones and zeros where ones represent abnormal
        frames and zeros represent normal
    """

    dataset = dataset_params['dataset']
    name = dataset_params['name']
    dataset_path = DATASETS[dataset][name]['DATASET_PATH']
    labels_path = DATASETS[dataset][name]['LABELS_PATH']
    test_num = dataset_params.get('test_num', 1)
    test_folder = os.path.join(dataset_path, "Test")
    ext = dataset_params['ext']

    labels_file = open(labels_path, 'r')
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


def get_labels_umn(dataset_params):
    """
    Function for getting labels of one particular dataset in test folder
    of the umn datasets.

    Parameters
    ----------
    dataset_params: dict
        dictionary of dataset parameters

    Returns
    -------
    labels: list
        list of ones and zeros where ones represent abnormal
        frames and zeros represent normal
    """

    dataset = dataset_params['dataset']
    name = dataset_params['name']
    dataset_path = DATASETS[dataset][name]['DATASET_PATH']
    labels_path = DATASETS[dataset][name]['LABELS_PATH']
    ext = dataset_params['ext']
    test_num = dataset_params['test_num']
    frames_folder = os.path.join(dataset_path, f'frames{test_num}')
    num_of_pics = len([x for x in os.listdir(frames_folder) if x.endswith(ext)])

    labels_file = open(labels_path, 'r')

    labels = []
    while True:

        line = labels_file.readline()

        if not line:
            break

        labels = [0] * num_of_pics

        parts = line.split(",")
        for p in parts:
            split = p.strip().split(":")
            lower, upper = int(split[0]), int(split[1])
            labels[lower - 1: upper] = [1] * (upper - lower + 1)
        break

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
    return dataset_function(dataset_params)


def labels_generator(dataset_params) -> Generator:
    """
    Generic function which generates labels for one particular dataset.

    Parameters
    ----------
    dataset_params: dict
        parameters needed for function which extracts labels

    Returns
    -------
    list_of_labels: list
        list of labels of each frame in SVOI
    """

    if dataset_params['dataset'] == UCSD:
        dataset_function = get_labels_ucsd
    else:
        dataset_function = get_labels_umn

    temporal_length = int(dataset_params.get('temporal_length', 7))

    labels = get_labels_from_dataset(dataset_function, dataset_params)
    current = 0

    while True:

        if current >= len(labels):
            return

        current_labels = labels[current: current + temporal_length]
        current += temporal_length

        yield current_labels


def normalize_cnn_output(output):
    """
    Normalizes values that come out of the cnn in a way that
    they represent probability of each class and sum to 1.

    Parameters
    ----------
    output: torch.tensor
        output of the cnn

    Returns
    -------
    output: torch.tensor
        probabilities of each class
    """

    out = output.data
    p1 = 1 / (1 + torch.exp(out[:, 1] - out[:, 0]))
    p2 = 1 / (1 + torch.exp(out[:, 0] - out[:, 1]))
    output = torch.stack([p1, p2], dim=1)
    output = torch.reshape(output, (out.shape[0], 2))
    return output


def get_dataset_and_frames_folders(dataset_params):
    """
    Function for getting folder path of the dataset based on the name of the dataset.

    Parameters
    ----------
    dataset_params: dict
        dataset parameters

    Returns
    -------
    folder_path, frames_folder: list, list
        folder of the dataset and folder with images of the dataset
    """

    dataset = dataset_params['dataset']

    dataset_name = dataset_params['name']

    if dataset == UCSD:
        if dataset_name == PED1:
            folder_path = UCSD_PED1_PATH
        else:
            folder_path = UCSD_PED2_PATH

        test_num = dataset_params.get('test_num', 1)
        frames_folder = os.path.join(folder_path, 'Test', 'Test{:03d}'.format(test_num))

    else:
        # UMN dataset
        if dataset_name == INDOOR:
            folder_path = UMN_INDOOR_PATH
        elif dataset_name == LAWN:
            folder_path = UMN_LAWN_PATH
        else:
            folder_path = UMN_PLAZA_PATH

        test_num = dataset_params.get('test_num', 1)
        frames_folder = os.path.join(folder_path, f'frames{test_num}')

    return folder_path, frames_folder


def train_and_test_indices(dataset_params: dict):
    """
    Function which returns folder indices used for training and testing.

    Parameters
    ----------
    dataset_params: dict
        dataset parameters in a form of a dictionary

    Returns
    -------
    train_folders, test_folders: list, list
        lists with indices for training folder and testing folder
    """

    dataset = dataset_params['dataset']
    name = dataset_params['name']
    training_set_size = dataset_params['training_set_size']

    if dataset == UCSD:

        path = os.path.join('data', dataset, name, 'Test')

        if dataset_params['only_gt']:

            gt_names = [x for x in os.listdir(path) if re.match('Test[0-9]{3}_gt$', x)]
            indexes = [re.findall('[0-9]{3}', x)[0] for x in gt_names]
            indexes = [x.lstrip('0') for x in indexes]
            indexes = [int(x) for x in indexes]

            total_gt_folders = len(indexes)
            num_of_folders_for_training = int(round(training_set_size * total_gt_folders))

            train_folders = set(indexes[:num_of_folders_for_training])
            test_folders = set(indexes).difference(train_folders)

            return list(train_folders), list(test_folders)

        else:
            number_of_folders = len([x for x in os.listdir(path) if re.match('Test[0-9]{3}$', x)])

    else:
        path = os.path.join('data', dataset, name)
        number_of_folders = len([x for x in os.listdir(path) if re.match('frames[0-9]$', x)])

    num_of_folders_for_training = int(round(training_set_size * number_of_folders))

    indexes = list(range(1, number_of_folders + 1))
    random.shuffle(indexes)

    train_folders = set(indexes[:num_of_folders_for_training])
    test_folders = set(indexes).difference(train_folders)

    return list(train_folders), list(test_folders)


def load_model(model_path: str):
    """
    Loads existing CNN model.

    Parameters
    ----------
    model_path: str
        path to existing model

    Returns
    -------
    model: CNN
        loaded model
    """

    assert os.path.exists(model_path), 'model path does not exists'
    print('loading model...')
    model = torch.load(model_path, map_location=torch.device('cpu'))
    print('model loaded')
    return model


def normalize_image(image, mean, std):
    """
    Normalizes image by subtracting mean of the dataset
    and dividing by standard deviation of the dataset.

    Parameters
    ----------
    image: np.ndarray
        image to normalize
    mean: float
        mean of the dataset
    std: float
        standard deviation of the dataset

    Returns
    -------
    normalized_image: np.ndarray
        normalized image
    """

    return (image - mean) / std


def save_model_data(dataset_params, svoi_params):
    """
    Saves all parameters used in training to a file in JSON format.

    Parameters
    ----------
    dataset_params: dict
        dataset parameters
    svoi_params: dict
        SVOI parameters
    """

    new_id = str(uuid.uuid1())

    with open(os.path.join(MODELS_DATA_PATH)) as json_file:
        data = json.load(json_file)
        temp = data['models']
        y = {
            'id': new_id,
            'dataset_parameters': {
                'dataset': dataset_params['dataset'],
                'name': dataset_params['name'],
                'training_set_size': dataset_params['training_set_size'],
                'ext': dataset_params['ext'],
                'epochs': dataset_params['epochs'],
                'batch_size': dataset_params['batch_size'],
                'train_indices': dataset_params['train_indices'],
                'test_indices': dataset_params['test_indices'],
                'learning_rate': dataset_params['lr'],
            },
            'SVOI parameters': {
                'temporal_length': svoi_params['temporal_length'],
                'square_size': svoi_params['square_size'],
                'resize_images': svoi_params['resize_images'],
                'sigma': svoi_params['sigma'],
                'p_s': svoi_params['p_s'],
            }
        }
        temp.append(y)

    with open(os.path.join(MODELS_DATA_PATH), 'w') as f:
        json.dump(data, f, indent=4)

    return new_id


def get_ground_truth_image_paths(dataset_params):
    """
    Function for getting ground truth images for UCSD dataset.

    Parameters
    ----------
    dataset_params: dict
        dataset parameters

    Returns
    -------
    gt_image_paths: List
        image paths
    """

    try:
        _, frames_folder = get_dataset_and_frames_folders(dataset_params)
        gt_path = frames_folder + '_gt'
        gt_images = [x for x in os.listdir(gt_path) if x.endswith(UCSD_GT_EXT)]
        gt_image_paths = [os.path.join(gt_path, x) for x in gt_images]
        return gt_image_paths

    except FileNotFoundError:
        return []


def test_models():
    """
    Function for evaluating available models in "models" folder.
    Writes results to "acc_file.txt" in folder "models".
    """

    acc_file = open(ACC_FILE_PATH, 'w')

    with open(MODELS_DATA_PATH) as json_file:
        data = json.load(json_file)
        models = data['models']
        for model_json in models:
            model_id = model_json['id']
            cnn = load_model(os.path.join(MODELS_PATH, model_id + '.pt'))
            cnn.dataset_params['device'] = 'cpu'
            acc = cnn.test()
            acc_file.write(model_id + ':' + str(acc) + '\n')

    acc_file.close()
