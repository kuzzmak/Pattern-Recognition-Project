from typing import Iterator, Iterable

import torch
from torch.utils.data import IterableDataset
import numpy as np

import util
from svoi import SVOI


class SVOIDataset(IterableDataset):

    def __init__(self, svoi_params: dict, dataset_params: dict):
        """
        Class representing iterable dataset where each iteration consists of SVOIs
        and corresponding labels for each SVOI.
        """

        self.dataset_params = dataset_params
        svoi_params['image_paths'] = self.get_image_paths()
        self.svoi = SVOI(svoi_params)

    def __iter__(self) -> Iterator:

        gt_path = util.get_ground_truth_image_paths(self.dataset_params)

        tl = self.dataset_params['temporal_length']
        # last svoi frame index
        frame_index = tl - 1
        batch_size = self.dataset_params['batch_size']
        # labels for current batch of SVOIs
        batch_labels = []
        # SVOIs for current batch
        batch_svois = []

        for s, labels in zip(self.svoi.generator(), util.labels_generator(self.dataset_params)):
            # go through all SVOIs, check if any frame of the ground truth frames
            # contain anomaly on the same place SVOI was extracted from original frames
            for index, (square, svoi) in enumerate(s.items()):
                # current svoi label
                label = 0
                # frames in SVOI contain abnormal event
                if 1 in labels:
                    # ground truth frames which are used to get real label of a SVOI
                    gt_frames = []
                    for i in range(frame_index, frame_index - tl, -1):
                        gt_frames.append(util.read_image(gt_path[i]))
                    gt_frames = np.stack(gt_frames, axis=0)

                    p1, p2 = square
                    svoi_gt = gt_frames[:, p1[0]:p2[0], p1[1]:p2[1]]

                    if 255 in svoi_gt:
                        label = 1

                resized_svoi = util.resize_svoi(svoi, (32, 32))
                svoi_tensor = util.make_tensor_from_svoi(resized_svoi)

                batch_svois.append(svoi_tensor)
                batch_labels.append(label)

                # yielding SVOIs and labels in batches of size batch_size
                if len(batch_svois) == batch_size:

                    batch_svois = torch.cat(batch_svois, dim=0)
                    targets = torch.tensor(np.array(batch_labels), dtype=torch.long)

                    yield batch_svois, targets

                    batch_svois = []
                    batch_labels = []

            frame_index += tl

        # if there are any number of SVOIs left less than batch size
        if len(batch_svois) > 0:

            batch_svois = torch.cat(batch_svois, dim=0)
            targets = torch.tensor(np.array(batch_labels), dtype=torch.long)

            yield batch_svois, targets

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
