import os
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.nn import Module
from torch.optim import optimizer

import util
from svoidataset import SVOIDataset


class CNN:

    def __init__(self, model: Module, optim: optimizer, criterion, save_model: bool, svoi_params: dict,
                 dataset_params: dict):
        """
        Class representing spatial temporal convolutional neural network.

        Parameters
        ----------
        model: Module
            base class for pytorch models
        optim
            which optimizer is used
        criterion:
            loss function
        save_model: bool
            should the model be saved
        svoi_params: dict
            parameters for SVOI
        dataset_params: dict
            dataset parameters
        """

        self.model = model
        self.optimizer = optim
        self.criterion = criterion
        self.save_model = save_model
        self.svoi_params = svoi_params
        self.dataset_params = dataset_params

    def train(self, epochs: int):
        """
        Method used for cnn training.

        Parameters
        ----------
        epochs: int
            number of iterations
        """

        self.model.train()

        for _ in tqdm(range(epochs), desc='Epoch: '):

            train_indexes, _ = util.train_and_test_indices(self.dataset_params)

            # for all folders that are used in training
            for index in tqdm(range(len(train_indexes)), desc='\tFolder: '):

                self.dataset_params['test_num'] = train_indexes[index]

                # make SVOIs and corresponding labels for current dataset folder
                sd = SVOIDataset(self.svoi_params, self.dataset_params)
                for s, labels in sd:

                    target = torch.tensor([int(1 in labels)], dtype=torch.long)

                    for square, svoi in s.items():
                        resized_svoi = util.resize_svoi(svoi, (32, 32))
                        svoi_tensor = util.make_tensor_from_svoi(resized_svoi)

                        output = self.model(svoi_tensor)
                        output = util.normalize_cnn_output(output)

                        loss = self.criterion(output, target)
                        loss = Variable(loss, requires_grad=True)

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

            # calculate errors
            # self.test()

        if self.save_model:
            print('saving model')
            new_id = util.save_model_data(self.dataset_params, self.svoi_params)
            model_path = os.path.join('models', f'{new_id}.pt')
            torch.save(self, model_path)
            print('model saved to: ', model_path)

    def test(self):
        """
        Used for evaluating classifier on new data.

        Returns
        -------
        error: float
            error percent
        """

        _, test_indexes = util.train_and_test_indices(self.dataset_params)
        for index in tqdm(range(len(test_indexes)), desc='\tFolder: '):

            self.dataset_params['test_num'] = test_indexes[index]

            # number of temporal_length frames which contain abnormal frame
            abnormal_frames_truth = 0
            # number of temporal_length frames which cnn classified as abnormal
            abnormal_frames_classified = 0

            sd = SVOIDataset(self.svoi_params, self.dataset_params)
            for s, labels in sd:

                target = torch.tensor([int(1 in labels)], dtype=torch.long)
                if target.item() == 1:
                    abnormal_frames_truth += 1

                for square, svoi in s.items():
                    resized_svoi = util.resize_svoi(svoi, (32, 32))
                    svoi_tensor = util.make_tensor_from_svoi(resized_svoi)

                    output = self.model(svoi_tensor)
                    output = util.normalize_cnn_output(output)

                    out = torch.argmax(output)
                    if out.item() == 1:
                        abnormal_frames_classified += 1
                        break

            print('Error: ')
            print(abs(abnormal_frames_truth - abnormal_frames_classified))
