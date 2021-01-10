from svoi2 import SVOI
import util
from cnn import CNN

import torch
import cv2 as cv
import numpy as np


def draw_str(dst: np.ndarray, text: str):
    """
    Function for writing centered text on image.

    Parameters
    ----------
    dst: np.ndarray
        image to which text is written
    text: str
        text which is written to image
    """

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_size = cv.getTextSize(text, font, font_scale, 2)[0]
    text_x = int((dst.shape[1] - text_size[0]) / 2)
    # text_y = int((dst.shape[0] + text_size[1]) / 2)
    text_y = 15
    cv.putText(dst, text, (text_x, text_y), font, font_scale, (255, 255, 255), 2)


class Classifier:

    def __init__(self, cnn: CNN, frames_folder: str, ext: str):
        """
        Class used for classifying new image frames with some cnn model.

        Parameters
        ----------
        cnn: CNN
            cnn model used for classifying
        frames_folder: str
            folder with frames used for classification
        ext: str
            extension of pictures in folder
        """

        self.cnn = cnn
        self.model = cnn.model
        self.frames_folder = frames_folder
        self.ext = ext

    def classify(self, autoplay=False):
        """
        Method which starts classification process.

        Parameters
        ----------
        autoplay: bool
            if frames should play by themselves
        """

        image_paths = util.get_image_paths(self.frames_folder, self.ext)
        svoi_params = self.cnn.svoi_params
        svoi_params['image_paths'] = image_paths

        sv = SVOI(svoi_params)
        temporal_length = svoi_params['temporal_length']
        i = temporal_length - 1
        for s in sv.generator():

            # image which is displayed
            im = util.read_image(image_paths[i])
            if svoi_params['resize_images']:
                image_shape = util.new_image_shape(im.shape, svoi_params['square_size'])
                im = util.resize_image(im, image_shape)

            i += temporal_length

            for square, svoi in s.items():

                resized_svoi = util.resize_svoi(svoi, (32, 32))
                svoi_tensor = util.make_tensor_from_svoi(resized_svoi)

                output = self.model(svoi_tensor)
                output = util.normalize_cnn_output(output)

                out = torch.argmax(output)
                if out.item() == 1:
                    draw_str(im, 'abnormal')
                    break

            cv.imshow('im', im)
            if autoplay:
                cv.waitKey(1)
            else:
                cv.waitKey(0)

