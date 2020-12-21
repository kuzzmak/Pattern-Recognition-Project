import numpy as np
import cv2 as cv
import os
import queue

# code taken from https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
from typing import Tuple

feature_params = dict(maxCorners=100,
                      qualityLevel=0.2,
                      minDistance=7,
                      blockSize=7)

lk_params = dict(winSize=(15, 15),
                 maxLevel=5,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

pic_num = 1
num_of_pics = 200

color = np.random.randint(0, 255, (100, 3))
# old_frame = cv.imread('data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001/001.tif')
# old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# x, y = 15, 15
# pic_shape1 = old_frame.shape
# step_x = int(pic_shape1[1] / x)
# step_y = int(pic_shape1[0] / y)

# for i in range(step_x):
#     for j in range(step_y):
#         start_point = (i * x, j * y)
#         end_point = ((i + 1) * y, (j + 1) * x)
#         # image = cv.rectangle(old_frame, start_point, end_point, (255, 0, 0), 1)
#         # cv.imshow('im', image)
#         # cv.waitKey(0)


def nearest_square(current: tuple, square_size: int) -> Tuple[tuple, tuple]:
    """
    Finds square area in which current pixel belongs to.

    :param current: pixel for which is area being searched for
    :param square_size: size of the square area
    :return: (y1, x1), (y2, x2) -> bounds of the square area
    """
    y_bound_up = int(square_size * np.ceil(current[0] / square_size))
    y_bound_down = y_bound_up - square_size
    x_bound_right = int(square_size * np.ceil(current[1] / square_size))
    x_bound_left = x_bound_right - square_size
    return (y_bound_down, x_bound_left), (y_bound_up, x_bound_right)


def add_square(svois, square_size, frame, dot):
    nearest = nearest_square(np.flipud(dot).ravel(), square_size)
    # upper left and lower right corner of the square area
    p1, p2 = nearest
    square_from_image = frame[p1[0]:p2[0], p1[1]:p2[1]]
    if square_from_image.shape == (square_size, square_size):
        if nearest not in svois:
            # no entry for this square yet
            svois[nearest] = [square_from_image]
        else:
            # add this frame to the existing frames
            svois[nearest].append(square_from_image)


def SVOI(path_to_images: str, temporal_length: int, square_size: int) -> dict:

    if not os.path.exists(path_to_images):
        raise FileNotFoundError(f'directory: {path_to_images} does not exists')

    images = [x for x in os.listdir(path_to_images) if x.endswith(".tif")]

    im_path = os.path.join(path_to_images, images[0])
    old_frame = cv.imread(im_path)
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

    # container for temporal_length images so every iteration of the algorithm
    # there is no IO operations on old images
    memory_images = queue.Queue(maxsize=temporal_length)
    memory_images.put(old_gray)
    # add another temporal_length - 1 images to the memory_images
    for i in range(temporal_length - 1):
        im_path = os.path.join(path_to_images, images[1 + i])
        memory_images.put(cv.cvtColor(cv.imread(im_path), cv.COLOR_BGR2GRAY))

    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # go through all images
    for i, im in enumerate(images, start=1):
        # if there are less frames left than temporal_length, stop generator
        if i + temporal_length > len(images):
            return

        # remove first image
        memory_images.get()
        temporal_images = list(memory_images.queue)

        # dictionary of regions(squares) of interest, if every square has
        # key of dictionary has temporal_length squares, it is considered
        # informative and thus returned by generator along with others
        # {key=square boundaries, value=list of temporal_length squares}
        svois = {}
        # extracting SVOI from temporal_length images
        for j, frame_gray in enumerate(temporal_images):
            # calculate optical flow
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for k, (new, old) in enumerate(zip(good_new, good_old)):
                if j == 0:
                    # squares of first image
                    add_square(svois, square_size, frame_gray, old)
                # squares from images that follow
                add_square(svois, square_size, frame_gray, new)

            # now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        im_path = os.path.join(path_to_images, images[i + temporal_length - 1])
        memory_images.put(cv.cvtColor(cv.imread(im_path), cv.COLOR_BGR2GRAY))

        svois_of_interest = {}
        for key, value in svois.items():
            if len(value) >= temporal_length:
                svois_of_interest[key] = value
        yield svois_of_interest


path = os.path.join('data', 'UCSD_Anomaly_Dataset.v1p2', 'UCSDped1', 'Train', 'Train001')
images = [x for x in os.listdir(path) if x.endswith(".tif")]
si = SVOI(path, 7, 15)

svois = next(si)
im0 = cv.imread(os.path.join(path, images[0]), 0)
for i in range(7):

    for key, value in svois.items():
        # invert x and y coordinates
        t1, t2 = key
        p1 = (t1[1], t1[0])
        p2 = (t2[1], t2[0])
        cv.rectangle(im0, p1, p2, (0, 255, 0), 1)
        # cv.imshow('im', im0)
        # cv.waitKey(0)
    cv.imshow('im', im0)
    cv.waitKey(0)
    im0 = cv.imread(os.path.join(path, images[i+1]), 0)
    svois = next(si)


# img = np.zeros((512, 512, 3), np.uint8)
# p1 = (75, 120)
# p2 = (90, 135)
# cv.rectangle(img, p1, p2, (0, 255, 0), 1)
# cv.imshow('im', img)
# cv.waitKey(0)
# n = next(si)
# n1 = next(si)
# n2 = next(si)
# print(len(n))
# print(len(n1))
# print(len(n2))

# old_frame = cv.imread(f'data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001/001.tif')
# old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
#
# p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# mask = np.zeros_like(old_frame)
#
# path = os.path.join('data', 'UCSD_Anomaly_Dataset.v1p2', 'UCSDped1', 'Train', 'Train001')
# images = [x for x in os.listdir(path) if x.endswith(".tif")]
#
# for pic in range(1, len(images)):
#     old_gray = cv.imread(os.path.join(path, images[pic]), 0)
#     p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
#     for dot in p0[:, 0]:
#         o = dot.ravel()
#         f = np.flipud(dot).ravel()
#         d = (dot[0], dot[1])
#         p1, p2 = nearest_square((dot[1], dot[0]), 15)
#         print(dot)
#
#         square_from_image = old_gray[p1[0]:p2[0], p1[1]:p2[1]]
#         if square_from_image.shape[0] < 2 or square_from_image.shape[1] < 2:
#             continue
#         print(square_from_image.shape)
#         cv.circle(old_frame, d, 3, (0, 0, 255), -1)
#         cv.rectangle(old_frame, p1, p1, (0, 0, 255), 1)
#         cv.imshow('im', old_frame)
#         cv.imshow('im2', square_from_image)
#         cv.waitKey(0)
#
#     cv.imshow('im', old_gray)
#     cv.waitKey(0)

# cv.imshow('im', old_gray)
# cv.waitKey(0)

# while 1:
#     pic_num += 1
#     if pic_num > num_of_pics:
#         break
#     frame = cv.imread(f'data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001/{pic_num:03d}.tif')
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # calculate optical flow
#     p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#     # Select good points
#     good_new = p1[st == 1]
#     good_old = p0[st == 1]
#     # draw the tracks
#     for i, (new, old) in enumerate(zip(good_new, good_old)):
#         a, b = new.ravel()
#         c, d = old.ravel()
#         nearest = nearest_square((a, b), 15)
#         mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
#         frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)
#     img = cv.add(frame, mask)
#     cv.imshow('frame', img)
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break
#     # Now update the previous frame and previous points
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1, 1, 2)



