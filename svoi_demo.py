import argparse
import os

from svoi import SVOI
import util

import cv2 as cv

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SVOI demo.')

    parser.add_argument('-f', type=str, default=os.path.join(util.UCSD_PED1_PATH, 'Test', 'Test001'),
                        metavar='--folder', help='folder to images')
    parser.add_argument('-ext', type=str, default='.tif', metavar='--extension', help='image extension')
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
    parser.add_argument('-resize-images', action='store_true', default=False,
                        help='resize input image to the multiple of square size (default: False)')

    args = parser.parse_args()

    image_paths = util.get_image_paths(args.f, args.ext)

    svoi_params = dict(
        resize_images=args.resize_images,
        temporal_length=args.tl,
        square_size=args.ss,
        image_paths=image_paths,
        sigma=args.sigma,
        p_s=args.ps,
    )

    frame_index = args.tl - 1
    sv = SVOI(svoi_params)
    for s in sv.generator():

        original = cv.imread(image_paths[frame_index], 0)
        frame_index += args.tl

        for square, svoi in s.items():
            p1, p2 = square
            cv.rectangle(original, (p1[1], p1[0]), (p2[1], p2[0]), (255, 0, 0), 1)

        cv.imshow('im', original)
        cv.waitKey(0)
