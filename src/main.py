import tkinter as tk
from tkinter import filedialog

import numpy as np
import cv2 as cv

from pathlib import Path
import os

from detect_utils import *


def translate_2d(image, tx, ty):
    """
    Shifts an image in the (x, y) direction by (tx, ty) and returns it.

    Args:
        image (numpy.ndarray): The input image.
        tx (float): The translation distance in the x-direction.
        ty (float): The translation distance in the y-direction.

    Returns:
        numpy.ndarray: The translated image.
    """
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    rows, cols = image.shape
    return cv.warpAffine(image, M, (cols, rows))


def align_images(inspected_img_path, reference_img_path):
    """
    Aligns the inspected image with the reference image using feature matching.

    Args:
        inspected_img_path (str): The file path of the inspected image.
        reference_img_path (str): The file path of the reference image.

    Returns:
        tuple: The translation distances (tx, ty) between the images.
    """
    insp_img = cv.imread(inspected_img_path, cv.IMREAD_GRAYSCALE)  # queryImage
    ref_img = cv.imread(reference_img_path, cv.IMREAD_GRAYSCALE)  # trainImage
    assert insp_img is not None, "align_images: Inspected image could not be read"
    assert ref_img is not None, "align_images: Reference image could not be read"

    # some blurring
    d = 9
    insp_img = cv.bilateralFilter(insp_img, d, d * 2, d / 2)
    ref_img = cv.bilateralFilter(ref_img, d, d * 2, d / 2)

    # initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(insp_img, None)
    kp2, des2 = orb.detectAndCompute(ref_img, None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # match descriptors.
    matches = bf.match(des1, des2)

    # sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # extract the matched keypoints from both images
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    # extract translation coordinates using the best match
    insp_pt1, ref_pt1 = src_pts[0], dst_pts[0]
    (tx, ty) = ref_pt1 - insp_pt1

    aligned_img = translate_2d(insp_img, tx, ty)

    # save the result
    cv.imwrite('./output/' + inspected_img_path.split('/')[-1], aligned_img)

    # create the corresponding reference image
    # compute the starting (x, y) coordinate of the ROI, ensuring that it is non-negative
    start_x = max(round(tx), 0)
    start_y = max(round(ty), 0)

    # compute the width and height of the region of interest, clipping it to the image boundaries
    w = ref_img.shape[1] - abs(round(tx))
    h = ref_img.shape[0] - abs(round(ty))

    # create a zero array with the same shape as the input image
    ref_roi_img = np.zeros_like(ref_img)

    # copy the region of interest from the input image to the zero array
    ref_roi_img[start_y:start_y+h, start_x:start_x +
                w] = ref_img[start_y:start_y+h, start_x:start_x+w]

    # save the resulting image
    cv.imwrite('./output/' + reference_img_path.split('/')[-1], ref_roi_img)

    return (tx, ty)


def detect(inspected_path, reference_path, translation):
    """
    Performs defect detection on the inspected image using the reference image.

    Args:
        inspected_path (str): The file path of the inspected image.
        reference_path (str): The file path of the reference image.
        translation (tuple): The translation distances (tx, ty) between the images.

    Returns:
        None
    """
    # load the images
    ref_img = cv.imread(reference_path, cv.IMREAD_GRAYSCALE)
    insp_img = cv.imread(inspected_path, cv.IMREAD_GRAYSCALE)
    assert ref_img is not None, "detect: Reference image could not be read."
    assert insp_img is not None, "detect: Inspected image could not be read."

    # subtract
    diff = cv.absdiff(ref_img, insp_img)

    # get defects by type
    res1 = find_rad(diff)
    res2 = find_rect(diff)
    res3 = find_hairs(diff)

    # aggregate types
    tmp = cv.bitwise_or(res1, res2)
    res = cv.bitwise_or(tmp, res3)

    # show & save result (if relevant)
    if is_image_all_zeros(res):
        print(f'[INFO] Could not find defects in this image.')
    else:
        # shift back to original position and save
        res_shifted_back = translate_2d(res, *(-t for t in translation))
        fname = inspected_path.split('/')[-1].split('.')[0]
        cv.imwrite('./output/' + fname + '_defects_mask.tif', res_shifted_back)

        # show results to user
        cv.imshow('Result', res)
        cv.imshow('Original (query) image', insp_img)
        print(f'[INFO] Displaying the result. Press any key to continue.')
        cv.waitKey(0)
        cv.destroyAllWindows()

    # remove temp files
    os.remove(inspected_path)
    os.remove(reference_path)


def main():
    """
    Main entry point of the defect detection application.
    """
    root_ = tk.Tk()
    root_.withdraw()

    while True:
        img_path_reference = filedialog.askopenfilename(
            title='Select the reference image'
        )
        img_path_inspected = filedialog.askopenfilename(
            title='Select the inspected image'
        )

        if img_path_reference and img_path_inspected:
            Path("./output").mkdir(parents=True, exist_ok=True)
            translation = align_images(img_path_inspected, img_path_reference)

            inspected_aligned_path = './output/' + \
                img_path_inspected.split('/')[-1]
            reference_aligned_path = './output/' + \
                img_path_reference.split('/')[-1]
            detect(inspected_aligned_path, reference_aligned_path, translation)
        else:
            print("[ERROR] You must provide two images.")

        if input("\nExamine another pair of images (Y/n)? ").lower() in ['n', 'no']:
            break


if __name__ == '__main__':
    main()
