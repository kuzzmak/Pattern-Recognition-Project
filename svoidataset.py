from typing import Iterator

from torch.utils.data import IterableDataset

import util
from svoi import SVOI


class SVOIDataset(IterableDataset):

    def __init__(self, dataset_path: str, ext: str, temporal_length: int, dataset_params: dict):
        """
        Class representing iterable dataset where each iteration consists of SVOI
        and corresponding labels for each frame of SVOI.

        How to use
        ----------
        dataset_path = os.path.join(util.UCSD_PATH, 'ped1', 'Test', 'Test001')
        sd = SVOIDataset(dataset_path, '.tif', 7, dataset_params)
        for svoi, labels in sd:
            do something...

        Parameters
        ----------
        dataset_path: str
            path to dataset folder with pictures
        ext: str
            extension of images in dataset
        temporal_length: int
            length of the SVOI in temporal domain
        dataset_params: dict
            parameters for dataset
        """

        self.image_paths = util.get_image_paths(dataset_path, ext)
        self.temporal_length = temporal_length
        self.dataset_params = dataset_params
        self.svoi = SVOI(self.image_paths)

    def __iter__(self) -> Iterator:
        return zip(
            self.svoi.generator(),
            util.labels_generator(
                self.temporal_length,
                util.get_labels_ucsd,
                self.dataset_params)
        )

    def __getitem__(self, index):
        pass
