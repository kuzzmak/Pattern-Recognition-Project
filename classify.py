import argparse
import os

from classifier import Classifier
import util

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify new image sequences with existing cnn model.')

    parser.add_argument('-m', type=str, default=os.path.join('models', 'cnn.pt'), metavar='--model',
                        help='path to cnn model')

    parser.add_argument('-f', type=str, default=os.path.join('data', 'UCSD', 'ped1', 'Test', 'Test001'),
                        metavar='--folder', help='folder with images')

    parser.add_argument('-ext', type=str, default='.tif', metavar='--extension', help='image extension')

    parser.add_argument('-a', action='store_true', default=False, help='should frames play normal or on user click')

    args = parser.parse_args()

    model_path = args.m
    folder_path = args.f
    ext = args.ext

    model = util.load_model(model_path)
    cl = Classifier(model, folder_path, ext)
    cl.classify(args.a)
