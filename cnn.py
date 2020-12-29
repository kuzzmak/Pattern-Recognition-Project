import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import util
from svoi import SVOI
from model import NetModel


class CNN:

    def __init__(self, model, optimizer, criterion, image_paths):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.image_paths = image_paths

    def train(self, epochs):
        self.model.train()

        for epoch in range(epochs):

            sv = SVOI(self.image_paths)
            for s in sv.generator():
                for square, svoi in s.items():
                    resized_svoi = util.resize_svoi(svoi, (32, 32))
                    svoi_tensor = util.make_tensor_from_svoi(resized_svoi)

                    output = self.model(svoi_tensor)
                    loss = self.criterion(output, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    print(output)

        # for batch_idx, (data, target) in enumerate(train_loader):
        #     data, target = data.to(device), target.to(device)
        #     self.optimizer.zero_grad()
        #     output = self.model(data)
        #     loss = F.nll_loss(output, target)
        #     loss.backward()
        #     self.optimizer.step()
        #
        #     if batch_idx % args.log_interval == 0:
        #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #             epoch, batch_idx * len(data), len(train_loader.dataset),
        #             100. * batch_idx / len(train_loader), loss.item()))
        #         if args.dry_run:
        #             break


if __name__ == "__main__":
    model = NetModel()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    folder_path = os.path.join('data', 'UCSD_Anomaly_Dataset.v1p2', 'UCSDped1', 'Train', 'Train001')
    image_paths = util.get_image_paths(folder_path, ".tif")
    cnn = CNN(model, optimizer, criterion, image_paths)

    epochs = 100
    cnn.train(epochs)
