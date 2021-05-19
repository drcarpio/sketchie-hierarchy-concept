import cv2
import numpy as np

'''
modified from solution on this page:
https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape
'''


def adjust_gamma(image, gamma=1.2):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


BLOCK_SIZE = 40
DELTA = 25


def preprocess(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return 255 - image


def get_block_index(image_shape, yx, block_size):
    y = np.arange(max(0, yx[0]-block_size),
                  min(image_shape[0], yx[0]+block_size))
    x = np.arange(max(0, yx[1]-block_size),
                  min(image_shape[1], yx[1]+block_size))
    return np.meshgrid(y, x)


def adaptive_median_threshold(img_in):
    med = np.median(img_in)
    img_out = np.zeros_like(img_in)
    img_out[img_in - med < DELTA] = 255
    kernel = np.ones((3, 3), np.uint8)
    img_out = 255 - cv2.dilate(255 - img_out, kernel, iterations=2)
    return img_out


def block_image_process(image, block_size):
    out_image = np.zeros_like(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            idx = (row, col)
            block_idx = get_block_index(image.shape, idx, block_size)
            out_image[block_idx] = adaptive_median_threshold(image[block_idx])
    return out_image


def process_image(img):
    image_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_in = preprocess(image_in)
    image_out = block_image_process(image_in, BLOCK_SIZE)
    return image_out


def adjust_image(img):
    adjust = adjust_gamma(img)
    process = process_image(adjust)
    ret, thresh = cv2.threshold(process, 127, 255, cv2.THRESH_BINARY_INV)
    dilate = cv2.dilate(thresh, None)
    erode = cv2.erode(dilate, None)
    return erode
