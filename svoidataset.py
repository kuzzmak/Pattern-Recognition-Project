from typing import Iterator, Iterable
import os

from torch.utils.data import IterableDataset

import util
from svoi import SVOI


class SVOIDataset(IterableDataset):

    def __init__(self, svoi_params: dict, dataset_params: dict):
        """
        Class representing iterable dataset where each iteration consists of SVOI
        and corresponding labels for each frame of SVOI.
        """

        self.dataset_params = dataset_params
        svoi_params['image_paths'] = self.get_image_paths()
        self.svoi = SVOI(svoi_params)

    def __iter__(self) -> Iterator:
        return zip(
            self.svoi.generator(),
            util.labels_generator(self.dataset_params)
        )

    def __getitem__(self, index):
        pass

    def get_image_paths(self) -> Iterable:
        """
        Function for getting path of all images in current folder
        so they can be easily loaded.

        Returns
        -------
        image_paths: list
            paths of images
        """

        _, frames_folder = util.get_dataset_and_frames_folders(self.dataset_params)
        image_paths = util.get_image_paths(frames_folder, self.dataset_params['ext'])
        return image_paths
