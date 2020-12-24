import numpy as np
import cv2 as cv
import os
import queue

# code taken from https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
from typing import Tuple

feature_params = dict(maxCorners=500,
                      qualityLevel=0.2,
                      minDistance=7,
                      blockSize=7)

lk_params = dict(winSize=(15, 15),
                 maxLevel=1,
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


# def SVOI(path_to_images: str, temporal_length: int, square_size: int) -> dict:
#     if not os.path.exists(path_to_images):
#         raise FileNotFoundError(f'directory: {path_to_images} does not exists')
#
#     images = [x for x in os.listdir(path_to_images) if x.endswith(".tif")]
#     assert len(images) >= temporal_length, f"there are no at least {temporal_length} images in folder {path_to_images}"
#
#     im_path = os.path.join(path_to_images, images[0])
#     old_frame = cv.imread(im_path)
#     old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
#
#     # container for temporal_length images so every iteration of the algorithm
#     # there is no IO operations on old images
#     memory_images = queue.Queue(maxsize=temporal_length)
#     memory_images.put(old_gray)
#     # add another temporal_length - 1 images to the memory_images
#     for i in range(temporal_length - 1):
#         im_path = os.path.join(path_to_images, images[1 + i])
#         memory_images.put(cv.cvtColor(cv.imread(im_path), cv.COLOR_BGR2GRAY))
#
#     p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
#
#     # go through all images
#     for i, im in enumerate(images, start=1):
#         print('in: ', str(i))
#         # if there are less frames left than temporal_length, stop generator
#         if i + temporal_length - 1 > len(images):
#             return
#
#         # remove first image
#         memory_images.get()
#         temporal_images = list(memory_images.queue)
#
#         # dictionary of regions(squares) of interest, if every square has
#         # key of dictionary has temporal_length squares, it is considered
#         # informative and thus returned by generator along with others
#         # {key=square boundaries, value=list of temporal_length squares}
#         svois = {}
#         # extracting SVOI from temporal_length images
#         for j, frame_gray in enumerate(temporal_images):
#             # calculate optical flow
#             p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#             # select good points
#             good_new = p1[st == 1]
#             good_old = p0[st == 1]
#
#             for k, (new, old) in enumerate(zip(good_new, good_old)):
#                 if j == 0:
#                     # squares of first image
#                     add_square(svois, square_size, frame_gray, old)
#                 # squares from images that follow
#                 add_square(svois, square_size, frame_gray, new)
#
#             # now update the previous frame and previous points
#             old_gray = frame_gray.copy()
#             p0 = good_new.reshape(-1, 1, 2)
#
#         if i + temporal_length - 1 < len(images):
#             im_path = os.path.join(path_to_images, images[i + temporal_length - 1])
#             memory_images.put(cv.cvtColor(cv.imread(im_path), cv.COLOR_BGR2GRAY))
#
#         svois_of_interest = {}
#         for k, v in svois.items():
#             if len(v) >= temporal_length:
#                 svois_of_interest[k] = v
#         yield svois_of_interest

path = os.path.join('data', 'UCSD_Anomaly_Dataset.v1p2', 'UCSDped1', 'Train', 'Train001')
images = [x for x in os.listdir(path) if x.endswith(".tif")]
first_frame = cv.imread(os.path.join(path, images[0]))
# first_frame = first_frame[0:15, 0:15, :]
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

##############################################3
# Farneback
#############################33
# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros_like(first_frame)
# Sets image saturation to maximum
mask[..., 1] = 255
temporal_length = 7
for i in range(0, len(images), temporal_length):
    prev_path = os.path.join(path, images[i])
    prev_gray = cv.imread(prev_path)
    for j in range(temporal_length):
        gray_path = os.path.join(path, images[i + j])
        gray = cv.imread(gray_path, 0)

        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        rgb_new = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

    # frame = cv.imread(os.path.join(path, images[i]))
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # mask[..., 0] = angle * 180 / np.pi / 2
    # mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    # rgb_new = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    # # cv.imshow("dense optical flow", rgb_new)
    #
    # sigma = 5
    # satisfies = rgb_new >= 15
    # seg = np.zeros_like(rgb_new)
    # seg[satisfies] = 255
    #
    #
    #
    # grey_3_channel = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    # # numpy_horizontal = np.hstack((rgb_new, grey_3_channel))
    #
    # rgb_old = rgb_new
    # # numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)
    # # numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
    # maximums = np.amax(seg, axis=2)
    # # cv2.imshow('Main', image)
    # # cv2.imshow('Numpy Vertical', numpy_vertical)
    #
    # gray_copy = gray.copy()
    # x, y = 15, 15
    # pic_shape1 = gray_copy.shape
    # step_x = int(pic_shape1[1] / x)
    # step_y = int(pic_shape1[0] / y)
    # test = seg[:, :, 0]
    # total_pixels_in_square = x * y
    # for i in range(step_y):
    #     for j in range(step_x):
    #         p1 = (i * x, j * y)
    #         p2 = ((i + 1) * y, (j + 1) * x)
    #         im_array = test[p1[0]:p2[0], p1[1]:p2[1]]
    #         greater = np.count_nonzero(im_array > 0)
    #         # print(greater, total_pixels_in_square * 0.5)
    #         if greater >= total_pixels_in_square * 0.65:
    #             # cv.imshow('little', im_array)
    #             print("nikkad")
    #             gray_copy = cv.rectangle(gray_copy, (p1[1], p1[0]), (p2[1], p2[0]), (255, 0, 0), 1)
    #             # cv.imshow('rect', gray_copy)
    #             # cv.waitKey(0)
    #
    # numpy_horizontal = np.hstack((gray_copy, seg[:, :, 0], seg[:, :, 1], seg[:, :, 2], maximums))
    # cv.imshow('Numpy Horizontal', numpy_horizontal)
    # # cv2.imshow('Numpy Vertical Concat', numpy_vertical_concat)
    # # cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)
    #
    # # cv.imshow("siva", cv.cvtColor(rgb_new, cv.COLOR_BGR2GRAY))
    # prev_gray = gray
    # if cv.waitKey(0) & 0xFF == ord('q'):
    #     break


######################################
# puno zelenih točaka
################################


# def draw_str(dst, target, s):
#     x, y = target
#     cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
#     cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)
#
#
# tracks = []
# track_len = 5
# detect_interval = 3
# frame_idx = 0
# while True:
#     if frame_idx > len(rgbs) - 1:
#         break
#     # frame = cv.imread(os.path.join(path, images[frame_idx]))
#     frame = rgbs[frame_idx]
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     vis = frame.copy()
#
#     if len(tracks) > 0:
#         img0, img1 = prev_gray, frame_gray
#         p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
#         p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
#         p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
#         d = abs(p0-p0r).reshape(-1, 2).max(-1)
#         good = d < 1
#         new_tracks = []
#         for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
#             if not good_flag:
#                 continue
#             tr.append((x, y))
#             if len(tr) > track_len:
#                 del tr[0]
#             new_tracks.append(tr)
#             p1, p2 = nearest_square((x, y), 15)
#             cv.rectangle(vis, p1, p2, (255, 0, 0), 1)
#             cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
#         tracks = new_tracks
#         # cv.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
#         # draw_str(vis, (20, 20), 'track count: %d' % len(tracks))
#
#     if frame_idx % detect_interval == 0:
#         mask = np.zeros_like(frame_gray)
#         mask[:] = 255
#         # for x, y in [np.int32(tr[-1]) for tr in tracks]:
#         #     cv.circle(mask, (x, y), 5, 0, -1)
#         p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
#         if p is not None:
#             for x, y in np.float32(p).reshape(-1, 2):
#                 tracks.append([(x, y)])
#
#     frame_idx += 1
#     prev_gray = frame_gray
#     cv.imshow('lk_track', vis)
#
#     ch = cv.waitKey(1)
#     if ch == 27:
#         break



# path = os.path.join('data', 'UCSD_Anomaly_Dataset.v1p2', 'UCSDped1', 'Train', 'Train001')
# images = [x for x in os.listdir(path) if x.endswith(".tif")]
# si = SVOI(path, 7, 15)
#
#
# for i, sv in enumerate(SVOI(path, 7, 15)):
#     im0 = cv.imread(os.path.join(path, images[i + 6]), 0)
#     for key, value in sv.items():
#         # invert x and y coordinates
#         t1, t2 = key
#         p1 = (t1[1], t1[0])
#         p2 = (t2[1], t2[0])
#         cv.rectangle(im0, p1, p2, (0, 255, 0), 1)
#     cv.imshow('im', im0)
#     cv.waitKey(1)
# svois = next(si)
# for i in range(6, len(images)):
#     print(i)
#     for key, value in svois.items():
#         # invert x and y coordinates
#         t1, t2 = key
#         p1 = (t1[1], t1[0])
#         p2 = (t2[1], t2[0])
#         cv.rectangle(im0, p1, p2, (0, 255, 0), 1)
#         # cv.imshow('im', im0)
#         # cv.waitKey(0)
#     cv.imshow('im', im0)
#     cv.waitKey(0)
#     # im0 = cv.imread(os.path.join(path, images[i+1]), 0)
#     svois = next(si)

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
