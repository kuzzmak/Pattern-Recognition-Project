import argparse

import torch
import torch.nn as nn

import util
from model import NetModel
from cnn import CNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training spatial temporal convolutional neural network')

    parser.add_argument('-e', type=int, default=20, metavar='--epochs', help='number of epochs to train (default: 20)')
    parser.add_argument('-lr', type=float, default=1.0, metavar='--learning-rate', help='learning rate (default: 1.0)')
    parser.add_argument('-tl', type=int, default=7, metavar='--temporal-length',
                        help='temporal length of the SVOI (default: 7)')
    parser.add_argument('-ss', type=int, default=15, metavar='--square-size',
                        help='square size of the SVOI (default: 15)')
    parser.add_argument('-sigma', type=float, default=10, metavar='--sigma',
                        help='levels of optical flow which need to be satisfied in '
                             'order to consider some square (default: 10)')
    parser.add_argument('-ps', type=float, default=0.5,
                        help='how many percent of pixels need to move inside '
                             'square so square is considered informative')
    parser.add_argument('-ts', type=float, default=0.7, metavar='--training_set_size',
                        help='how many percent of data is used for training')
    parser.add_argument('-resize-images', action='store_true', default=False,
                        help='resize input image to the multiple of square size (default: False)')
    parser.add_argument('-ext', type=str, metavar='--extension', default='.tif',
                        help='image extension (default: .tif)')
    parser.add_argument('-save-model', action='store_true', default=False,
                        help='for saving the current model')
    parser.add_argument('-dataset', type=str, choices=[util.UCSD, util.UMN], default=util.UCSD,
                        help='which dataset (default: UCSD)')
    parser.add_argument('-dataset-name', type=str, choices=[util.PED1, util.PED2, util.LAWN, util.PLAZA, util.INDOOR],
                        default=util.PED1, help='which dataset name (default: PED1)')
    parser.add_argument('-only-gt', action='store_true', default=False,
                        help='use only ground truth directories (default: False, possible only with UCSD)')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_params = dict(
        training_set_size=args.ts,
        dataset=args.dataset,
        name=args.dataset_name,
        temporal_length=args.tl,
        only_gt=args.only_gt,
        device=device,
        batch_size=20,
        epochs=args.e,
        ext=args.ext,
        lr=args.lr,
    )
    svoi_params = dict(
        resize_images=args.resize_images,
        temporal_length=args.tl,
        square_size=args.ss,
        sigma=args.sigma,
        p_s=args.ps,
    )

    model = NetModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    cnn = CNN(model, optimizer, criterion, True, svoi_params, dataset_params)
    cnn.train()
