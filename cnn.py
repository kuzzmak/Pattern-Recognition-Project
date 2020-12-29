import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output


class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()

        # Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3, out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=64, out_channels=128)
        self.unit9 = Unit(in_channels=128, out_channels=128)
        self.unit10 = Unit(in_channels=128, out_channels=128)
        self.unit11 = Unit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=128, out_channels=128)
        self.unit13 = Unit(in_channels=128, out_channels=128)
        self.unit14 = Unit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        # Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
                                 , self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgpool)

        self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 128)
        output = self.fc(output)
        return output


# Define transformations for the training set, flip the images randomly, crop out and apply mean and std normalization
train_transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 10

# Load the training set
train_set = CIFAR10(root="./data", train=True, transform=train_transformations)

# Create a loder for the training set
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

# Define transformations for the test set
test_transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])

# Load the test set, note that train is set to False
test_set = CIFAR10(root="./data", train=False, transform=test_transformations)

# Create a loder for the test set, note that both shuffle is set to false for the test loader
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

# Check if gpu support is available
cuda_avail = torch.cuda.is_available()

# Create model, optimizer and loss function
model = SimpleNet(num_classes=10)

if cuda_avail:
    model.cuda()

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()


# Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
def adjust_learning_rate(epoch):
    lr = 0.001

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_models(epoch):
    torch.save(model.state_dict(), "cifar10model_{}.model".format(epoch))
    print("Checkpoint saved")


def test():
    model.eval()
    test_acc = 0.0
    for i, (images, labels) in enumerate(test_loader):

        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        # Predict classes using images from the test set
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        prediction = prediction.cpu().numpy()
        test_acc += torch.sum(prediction == labels.data)

    # Compute the average acc and loss over all 10000 test images
    test_acc = test_acc / 10000

    return test_acc


def train(num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Move images and labels to gpu if available
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            # Clear all accumulated gradients
            optimizer.zero_grad()
            # Predict classes using images from the test set
            outputs = model(images)
            # Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs, labels)
            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.cpu().data * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_acc += torch.sum(prediction == labels.data)

        # Call the learning rate adjustment function
        adjust_learning_rate(epoch)

        # Compute the average acc and loss over all 50000 training images
        train_acc = train_acc / 50000
        train_loss = train_loss / 50000

        # Evaluate on the test set
        test_acc = test()

        # Save the model if the test acc is greater than our current best
        if test_acc > best_acc:
            save_models(epoch)
            best_acc = test_acc

        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc, train_loss,
                                                                                        test_acc))


import util
import cv2 as cv
from svoi import SVOI
import os


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.c1 = nn.Conv3d(in_channels=1, out_channels=12, kernel_size=3)  # out [1, 12, 5, 30, 30]
        self.c2 = nn.Conv3d(in_channels=12, out_channels=24, kernel_size=(1, 3, 3))  # out [1, 24, 5, 28, 28]
        self.c3 = nn.Conv3d(in_channels=24, out_channels=48, kernel_size=(3, 3, 3))  # out [1, 48, 3, 12, 12]
        self.c4 = nn.Conv3d(in_channels=48, out_channels=64, kernel_size=(3, 3, 3))  # out [1, 64, 1, 4, 4]
        self.c5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 4, 4))  # out [1, 64, 1, 1, 1]
        self.l1 = nn.Linear(in_features=64, out_features=128)  # out [1, 128]
        self.l2 = nn.Linear(in_features=128, out_features=2)  # out [1, 2]
        self.pol = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.net = nn.Sequential(
            self.c1,
            self.c2,
            self.pol,
            self.c3,
            self.pol,
            self.c4,
            self.c5,
        )

    def forward(self, x):
        out = self.net(x)
        out = self.l1(out.view(-1, 64))
        out = self.l2(out)
        print(out.shape)
        return out


if __name__ == "__main__":
    # train(200)
    # folder_path = os.path.join('data', 'UCSD_Anomaly_Dataset.v1p2', 'UCSDped1', 'Train', 'Train001')
    # image_paths = util.get_image_paths(folder_path, ".tif")
    # image_shape = cv.imread(image_paths[0]).shape
    #
    # new_shape = util.new_image_shape(image_shape, 15)
    # print(image_shape)
    # print(new_shape)
    #
    # sv = SVOI(image_paths)
    # i = 6
    # for s in sv.generator():
    #     im = cv.imread(image_paths[i], 0)
    #     black = np.zeros_like(im)
    #     i += 7
    #     for square, svoi in s.items():
    #         p1, p2 = square
    #         cv.rectangle(im, (p1[1], p1[0]), (p2[1], p2[0]), (255, 0, 0), 1)
    #         black[p1[0]:p2[0], p1[1]:p2[1]] = svoi[:, :, 6]
    #     cv.imshow('im', im)
    #     cv.imshow('black', black)
    #     cv.waitKey(0)

    folder_path = os.path.join('data', 'UCSD_Anomaly_Dataset.v1p2', 'UCSDped1', 'Train', 'Train001')
    image_paths = util.get_image_paths(folder_path, ".tif")
    sv = SVOI(image_paths)
    old_svoi = []
    for s in sv.generator():
        for square, svoi in s.items():
            old_svoi = svoi
            break
        break

    # normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # c1 = nn.Conv3d(in_channels=1, out_channels=12, kernel_size=(3, 3, 3))
    # c2 = nn.Conv3d(in_channels=12, out_channels=24, kernel_size=(1, 3, 3))
    # c3 = nn.Conv3d(in_channels=24, out_channels=48, kernel_size=(3, 3, 3))
    # c4 = nn.Conv3d(in_channels=48, out_channels=64, kernel_size=(3, 3, 3))
    # c5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 4, 4))
    # l = nn.Linear(in_features=64, out_features=128)

    resized_svoi = util.resize_svoi(old_svoi, (32, 32))
    ten = util.make_tensor_from_svoi(resized_svoi)

    net = Net()
    out = net.forward(ten)
    print(out.shape)
    # out_c1 = c1(ten)
    # print(out_c1.shape)
    # # out_c1 = torch.randn(1, 12, 30, 30)
    # out_c2 = c2(out_c1)
    # print(out_c2.shape)
    #
    # m = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
    # pol = m(out_c2)
    # print("pol")
    # print(pol.shape)
    #
    # out_c3 = c3(pol)
    # print(out_c3.shape)
    #
    # m = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
    # pol = m(out_c3)
    # print("pol")
    # print(pol.shape)
    #
    # out_c4 = c4(pol)
    # print(out_c4.shape)
    #
    # out_c5 = c5(out_c4)
    # print(out_c5.shape)
    #
    # fc_out = l(out_c5.view(-1, 64))
    # print(fc_out.shape)

    # a = c1(ten)
    # print(a.shape)
    # print(a)
    # svoi = np.random.randint(0, 255, (7, 15, 15))
    # svoi_resized = util.resize_svoi(svoi)
    # a = torch.randn(1, 7, 7, 15, 15)
    # svoi = np.random.randint()
    # f = cv.resize(d, (7, 32, 32))
    # print(f.shape)
    # b = torch.randn(1, 3, 32, 32)
    #
    # c = u(b)
    #
    # print(c)
    # train(200)
    # g = u1(a)
    # print(g.shape)
    # a = nn.Conv3d(in_channels=7, out_channels=12, kernel_size=3)
    # b = torch.randn(15, 15, 7)
    # c = a(b)
    # print(a)
