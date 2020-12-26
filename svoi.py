import numpy as np
import cv2 as cv

import util


class SVOI:

    def __init__(self, image_paths, resize_images=False, temporal_length=7, square_size=15):
        """
        Class used for extracting spatial-temporal volumes of interest (SVOI).
        SVOI consists of temporal_length frames in time domain. Although it's
        easy to extract all SVOIs from frames, here are extracted only "useful"
        SVOIs. Useful SVOI is SVOI in which at least p_s percent of pixels are
        moving. For moving pixels Farneback method for deep optical flow is used.

        :param image_paths: paths of all images used in SVOI extraction
        :param resize_images: resize images in a way that each size is the multiple of the square size
        :param temporal_length: how many frames in time domain is used in SVOI
        :param square_size: size of the square into which image is divided
        """

        self.image_paths = image_paths
        self.resize_images = resize_images
        self.temporal_length = temporal_length
        self.square_size = square_size
        # shape of every image used in SVOI extraction
        self.image_shape = cv.imread(self.image_paths[0]).shape
        # change image shape if resizing is used
        if resize_images:
            self.image_shape = util.new_image_shape(self.image_shape, self.square_size)
        # creates an image filled with zero intensities with the same dimensions as the frame
        self.mask = np.zeros(self.image_shape, dtype='uint8')
        # ets image saturation to maximum
        self.mask[..., 1] = 255
        # total pixels in one frame of a SVOI
        self.total_pixels_in_square = square_size * square_size
        # pixel intensities greater or equal to sigma must be in
        # newly made rgb image which shows deep optical flow
        self.sigma = 10
        # percent of pixels in SVOI square that need to change
        # (from one frame to another) to be considered useful
        self.p_s = 0.5
        # smallest number of squares which need to be present in
        # temporal_length frames in SVOI
        self.num_of_squares = 3
        # how many steps in x direction of the image when going square by square
        self.step_x = int(self.image_shape[1] / square_size)
        self.step_y = int(self.image_shape[0] / square_size)

        self.farneback_params = dict(pyr_scale=0.5,
                                     levels=3,
                                     winsize=15,
                                     iterations=5,
                                     poly_n=7,
                                     poly_sigma=1.5,
                                     flags=0)

    def count_incidence_od_squares(self, incidence_of_squares, seg):

        for i in range(self.step_y):
            for j in range(self.step_x):

                p1 = (i * self.square_size, j * self.square_size)
                p2 = ((i + 1) * self.square_size, (j + 1) * self.square_size)

                im_array = seg[p1[0]:p2[0], p1[1]:p2[1]]

                greater = np.count_nonzero(im_array > 0)
                if greater >= self.total_pixels_in_square * self.p_s:
                    square = (p1, p2)
                    if square not in incidence_of_squares:
                        incidence_of_squares[square] = 1
                    else:
                        incidence_of_squares[square] += 1

    def count_total_incidence_by_frame(self, incidence_of_squares_by_frame, seg):

        for k, v in incidence_of_squares_by_frame.items():
            self.count_incidence_od_squares(incidence_of_squares_by_frame[k], seg[:, :, k])

    def count_total_incidence_by_channel(self, incidence_of_squares_by_frame):
        incidence_of_squares_total = {}
        for channel in range(self.image_shape[2]):
            for k, v in incidence_of_squares_by_frame[channel].items():
                if v >= self.num_of_squares:
                    if k not in incidence_of_squares_total:
                        incidence_of_squares_total[k] = 1
                    else:
                        incidence_of_squares_total[k] += 1
        return incidence_of_squares_total

    def get_svois_from_frames(self, current_frames, incidence_of_squares_total):
        svois = {}
        for k, v in incidence_of_squares_total.items():
            # image has 3 channels, at least two have to contain square
            if v >= 2:
                svoi = np.zeros((self.square_size, self.square_size, self.temporal_length))
                p1, p2 = k
                for i, frame in enumerate(current_frames):
                    square = frame[p1[0]:p2[0], p1[1]:p2[1]]
                    svoi[:, :, i] = square
                svois[k] = svoi
        return svois

    def generator(self):

        for i in range(0, len(self.image_paths), self.temporal_length):

            if i + self.temporal_length > len(self.image_paths):
                return

            # temporal_length of frames used for SVOI extraction
            current_frames = []
            # first image of temporal_length frames
            prev_gray = util.read_image(self.image_paths[i])
            if self.resize_images:
                prev_gray = util.resize_image(prev_gray, self.image_shape)
            current_frames.append(prev_gray)

            incidence_of_squares_by_frame = {}
            for j in range(self.image_shape[2]):
                incidence_of_squares_by_frame[j] = {}

            for j in range(1, self.temporal_length):
                # remaining temporal_length - 1 frames
                gray = util.read_image(self.image_paths[i + j])
                if self.resize_images:
                    prev_gray = util.resize_image(prev_gray, self.image_shape)
                current_frames.append(gray)

                flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, **self.farneback_params)
                magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
                self.mask[..., 0] = angle * 180 / np.pi / 2
                self.mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
                rgb_new = cv.cvtColor(self.mask, cv.COLOR_HSV2BGR)

                # pixels that satisfy condition, pixels which intensities
                # are greater than some threshold
                satisfies = rgb_new >= self.sigma
                seg = np.zeros_like(rgb_new)
                # array like satisfies which has 255 intensity on all
                # the places which satisfy threshold condition
                seg[satisfies] = 255

                self.count_total_incidence_by_frame(incidence_of_squares_by_frame, seg)

                prev_gray = gray

            incidence_of_squares_total = self.count_total_incidence_by_channel(incidence_of_squares_by_frame)
            svois = self.get_svois_from_frames(current_frames, incidence_of_squares_total)

            yield svois
