import argparse
import os

import torch
import torch.nn as nn

import util
from model import NetModel
from cnn import CNN


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Spatial-temporal Convolutional Neural Networks for '
                                                 'Anomaly Detection and Localization in Crowded Scenes.')

    parser.add_argument('-e', type=int, default=20, metavar='--epochs', help='number of epochs to train (default: 20)')
    parser.add_argument('-lr', type=float, default=1.0, metavar='--learning-rate', help='learning rate (default: 1.0)')
    parser.add_argument('-tl', type=int, default=7, metavar='--temporal-length',
                        help='temporal length of the SVOI (default: 7)')
    parser.add_argument('-ss', type=int, default=15, metavar='--square-size',
                        help='square size of the SVOI (default: 15)')
    parser.add_argument('-resize-images', action='store_true', default=False,
                        help='resize input image to the multiple of square size (default: False)')
    parser.add_argument('-ext', type=str, metavar='--extension', default='.tif',
                        help='image extension (default: .tif)')
    parser.add_argument('-save-model', action='store_true', default=False,
                        help='for saving the current model')
    parser.add_argument('-dataset', type=str, choices=[util.UCSD, util.UMN], default=util.UCSD,
                        help='which dataset (default: UCSD)')
    parser.add_argument('-dataset_name', type=str, choices=[util.PED1, util.PED2, util.LAWN, util.PLAZA, util.INDOOR],
                        default=util.PED1, help='which dataset name (default: PED1)')
    parser.add_argument('-num', type=int, default=1, help='which test in UCSD dataset is used to train (default: 1)')

    args = parser.parse_args()

    dataset_params = dict(
        dataset=args.dataset,
        name=args.dataset_name,
        test_num=args.num,
        temporal_length=args.tl,
        ext=args.ext,
    )

    folder_path = os.path.join('data', 'UCSD', 'ped1', 'Test', 'Test001')
    image_paths = util.get_image_paths(folder_path, args.ext)
    svoi_params = dict(
        image_paths=image_paths,
        resize_images=args.resize_images,
        temporal_length=args.tl,
        square_size=args.ss,
        sigma=10,
        p_s=0.5,
    )

    model = NetModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    cnn = CNN(model, optimizer, criterion, args.save_model, svoi_params, dataset_params)
    cnn.train(args.e)
