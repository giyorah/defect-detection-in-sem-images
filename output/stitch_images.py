import cv2 as cv
import numpy as np

# Load the two images to be stitched
for i in range(1, 4):
    img_left = cv.imread(f'case{i}_inspected_image.tif')
    img_right = cv.imread(f'case{i}_inspected_image_defects_mask.tif')

    # Check that the height of the images is the same
    assert img_left.shape[0] == img_right.shape[0]

    # Create a new image with a width equal to the sum of the widths of the two images
    img_stitched = np.zeros((img_left.shape[0], img_left.shape[1] + img_right.shape[1], 3), dtype=np.uint8)

    # Copy the pixels from the first image into the left half of the new image
    img_stitched[:, :img_left.shape[1], :] = img_left

    # Copy the pixels from the second image into the right half of the new image, starting at the column that corresponds to the width of the first image
    img_stitched[:, img_left.shape[1]:, :] = img_right

    # Display or save the resulting image
    cv.imshow(f'Case{i} - Stitched images', img_stitched)
    cv.waitKey(0)
    
cv.destroyAllWindows()
