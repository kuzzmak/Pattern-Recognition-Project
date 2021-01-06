import os

import torch
from torch.autograd import Variable

import util
from svoidataset import SVOIDataset


class CNN:

    def __init__(self, model, optimizer, criterion, save_model, svoi_params, dataset_params):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_model = save_model
        self.svoi_params = svoi_params
        self.dataset_params = dataset_params

    @staticmethod
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

        out = output.data[0]
        p1 = 1 / (1 + torch.exp(out[1] - out[0]))
        p2 = 1 / (1 + torch.exp(out[0] - out[1]))
        return torch.tensor([[p1, p2]], dtype=torch.float32)

    def train(self, epochs):
        self.model.train()
        print('training started...')

        for epoch in range(epochs):

            sd = SVOIDataset(self.svoi_params, self.dataset_params)
            for s, labels in sd:

                target = torch.tensor([int(1 in labels)], dtype=torch.long)

                for square, svoi in s.items():
                    resized_svoi = util.resize_svoi(svoi, (32, 32))
                    svoi_tensor = util.make_tensor_from_svoi(resized_svoi)

                    output = self.model(svoi_tensor)
                    output = self.normalize_cnn_output(output)

                    loss = self.criterion(output, target)
                    loss = Variable(loss, requires_grad=True)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            print(f'\tepoch: {epoch + 1}')
        print('training finished')

        if self.save_model:
            print('saving model')
            model_path = os.path.join('models', 'cnn.pt')
            torch.save(self, model_path)
            print('model saved to: ', model_path)
