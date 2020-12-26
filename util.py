import cv2 as cv


def read_image(path, flag=cv.IMREAD_GRAYSCALE):
    return cv.imread(path, flag)