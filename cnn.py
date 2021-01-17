import os
import random

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

    def train(self):
        """
        Method used for cnn training.
        """

        self.model.train()

        train_indices, test_indices = util.train_and_test_indices(self.dataset_params)
        self.dataset_params['train_indices'] = train_indices
        self.dataset_params['test_indices'] = test_indices

        device = self.dataset_params['device']
        epochs = self.dataset_params['epochs']

        for _ in tqdm(range(epochs), desc='Epoch: '):

            random.shuffle(train_indices)

            # for all folders that are used in training
            for index in tqdm(range(len(train_indices)), desc='\tTrain Folder: ', leave=False):

                self.dataset_params['test_num'] = train_indices[index]

                # make SVOIs and corresponding labels for current dataset folder
                sd = SVOIDataset(self.svoi_params, self.dataset_params)
                for svois, targets in sd:

                    targets = targets.to(device)
                    inputs = svois.to(device)

                    output = self.model(inputs)
                    output = util.normalize_cnn_output(output)
                    output = output.to(device)

                    loss = self.criterion(output, targets)
                    loss = Variable(loss, requires_grad=True)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # acc = self.test(test_indices)
            # tqdm.write(f'acc: {acc}')

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

        # self.model.eval()

        correct = 0
        total = 0

        device = self.dataset_params['device']

        with torch.no_grad():

            for index in tqdm(range(len(self.dataset_params['train_indices'])), desc='\tTest Folder: '):

                self.dataset_params['test_num'] = self.dataset_params['train_indices'][index]

                sd = SVOIDataset(self.svoi_params, self.dataset_params)
                for svois, targets in sd:

                    total += svois.shape[0]

                    targets = targets.to(device)
                    inputs = svois.to(device)

                    output = self.model(inputs)
                    output = util.normalize_cnn_output(output)
                    output = output.to(device)

                    _, predicted = torch.max(output.data, 1)
                    correct += (predicted == targets).sum().item()

        return correct / total
