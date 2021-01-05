import torch
import torch.nn as nn

import numpy as np

import util
from model import NetModel
from svoidataset import SVOIDataset


class CNN:

    def __init__(self, model, optimizer, criterion, dataset_params):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataset_params = dataset_params

    def train(self, epochs):
        self.model.train()

        for epoch in range(epochs):

            sd = SVOIDataset(self.dataset_params)
            for s, labels in sd:

                target = int(1 in labels)

                for square, svoi in s.items():
                    resized_svoi = util.resize_svoi(svoi, (32, 32))
                    svoi_tensor = util.make_tensor_from_svoi(resized_svoi)

                    # TODO fix this
                    # output = self.model(svoi_tensor)
                    # probs = util.get_probabilities_from_cnn_output(output.data[0])
                    # output = torch.tensor([[np.argmax(probs)]], dtype=torch.float32)
                    # output = np.argmax(probs)
                    # loss = self.criterion(output, target)
                    #
                    # self.optimizer.zero_grad()
                    # loss.backward()
                    # self.optimizer.step()

            print(f'epoch: {epoch}')


if __name__ == "__main__":
    _model = NetModel()

    _optimizer = torch.optim.SGD(_model.parameters(), lr=0.1)
    _criterion = nn.CrossEntropyLoss()

    _dataset_params_ucsd = dict(
        dataset=util.UCSD,
        name=util.PED1,
        test_num=1,
        temporal_length=7,
        ext='.tif',
    )
    cnn = CNN(_model, _optimizer, _criterion, _dataset_params_ucsd)

    _epochs = 100
    cnn.train(_epochs)
