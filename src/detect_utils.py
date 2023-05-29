import cv2 as cv
import numpy as np


def find_hairs(diff_img):
    """
    Finds "hair-like" defects in the given difference image.

    Args:
        diff_img (numpy.ndarray): The difference image to analyze.

    Returns:
        numpy.ndarray: A binary image highlighting the "hair-like" defects.
    """

    # apply a threshold to get a binary image
    _, thresholded_diff = cv.threshold(diff_img, 17, 255, cv.THRESH_BINARY)

    # find contour boundary of white regions
    contours, _ = cv.findContours(
        thresholded_diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_image = np.zeros_like(thresholded_diff)
    for i, contour in enumerate(contours):
        hull = cv.convexHull(contour)
        hull_area = cv.contourArea(hull)

        if 100 < hull_area < 1000:
            contour_area = cv.contourArea(contour)
            solidity = float(contour_area) / hull_area
            if solidity < 0.05:
                # draw the "hair"
                cv.drawContours(contours_image, contours, i, (255, 255, 255))

    return contours_image


def find_rect(diff_img):
    """
    Finds rectangular defects in the given difference image.

    Args:
        diff_img (numpy.ndarray): The difference image to analyze.

    Returns:
        numpy.ndarray: A binary image highlighting the rectangular defects.
    """

    # apply a threshold to get a binary image
    _, thresholded_diff = cv.threshold(diff_img, 60, 255, cv.THRESH_BINARY)

    # erode & dilate
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
    opened = cv.morphologyEx(thresholded_diff, cv.MORPH_OPEN, kernel)

    return opened


def find_rad(diff_img):
    """
    Finds radial blotches in the given difference image.  

    Args:
        diff_img (numpy.ndarray): The difference image to analyze.  

    Returns:
        numpy.ndarray: A binary image highlighting the radial blotches.
    """

    # apply Gaussian blur
    image = cv.GaussianBlur(diff_img, (0, 0), 3)

    # sharpen the difference image
    sharp_diff = cv.addWeighted(diff_img, 1.6, image, -0.6, 0)

    # apply a threshold to get a binary image
    _, thresholded_diff = cv.threshold(sharp_diff, 20, 255, cv.THRESH_BINARY)

    # erode & dilate
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    opened = cv.morphologyEx(thresholded_diff, cv.MORPH_OPEN, kernel)

    return opened


def is_image_all_zeros(img):
    """
    Returns True if all pixels in the input image are black (i.e. have a value of zero), False otherwise.
    """
    return cv.countNonZero(img) == 0
