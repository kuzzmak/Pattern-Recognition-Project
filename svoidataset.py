from typing import Iterator, Iterable
import os

from torch.utils.data import IterableDataset

import util
from svoi import SVOI


class SVOIDataset(IterableDataset):

    def __init__(self, dataset_params: dict):
        """
        Class representing iterable dataset where each iteration consists of SVOI
        and corresponding labels for each frame of SVOI.
        """

        self.dataset_params = dataset_params
        self.image_paths = self.get_image_paths(dataset_params)
        self.svoi = SVOI(self.image_paths)

    def __iter__(self) -> Iterator:
        return zip(
            self.svoi.generator(),
            util.labels_generator(self.dataset_params)
        )

    def __getitem__(self, index):
        pass

    @staticmethod
    def get_image_paths(dataset_params: dict) -> Iterable:

        dataset = dataset_params['dataset']

        dataset_name = dataset_params['name']
        ext = dataset_params['ext']
        if dataset == util.UCSD:
            if dataset_name == util.PED1:
                folder_path = util.UCSD_PED1_PATH
            else:
                folder_path = util.UCSD_PED2_PATH

            test_num = dataset_params.get('test_num', 1)
            frames_folder = os.path.join(folder_path, 'Test', 'Test{:03d}'.format(test_num))

        else:
            # UMN dataset
            if dataset_name == util.INDOOR:
                folder_path = util.UMN_INDOOR_PATH
            elif dataset_name == util.LAWN:
                folder_path = util.UMN_LAWN_PATH
            else:
                folder_path = util.UMN_PLAZA_PATH

            frames_folder = os.path.join(folder_path, 'frames')

        image_paths = util.get_image_paths(frames_folder, ext)

        return image_paths
