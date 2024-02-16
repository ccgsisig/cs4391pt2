"""
CS 4391 Homework 2 Programming
Implement the harris_corner() function and the non_maximum_suppression() function in this python script
Harris corner detector
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def non_maximum_suppression(R):
    mask = np.zeros_like(R, dtype=np.uint8)
    height, width = R.shape

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Check 8 neighbors
            neighbors = R[i-1:i+2, j-1:j+2].flatten()
            neighbors[4] = 0  # Exclude the center pixel
            if R[i, j] > np.max(neighbors):
                mask[i, j] = 1

    return mask


def harris_corner(im):
    # Convert RGB to gray-scale image
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)/255

    # Compute image gradient using Sobel filters
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute products of derivatives at every pixel
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy

    # Compute the sums of products of derivatives at each pixel using Gaussian filter
    sigma = 1.5
    ksize = int(6 * sigma) | 1
    Mxx = cv2.GaussianBlur(Ixx, (ksize, ksize), sigma)
    Mxy = cv2.GaussianBlur(Ixy, (ksize, ksize), sigma)
    Myy = cv2.GaussianBlur(Iyy, (ksize, ksize), sigma)

    # Compute determinant and trace of the M matrix
    det_M = Mxx * Myy - Mxy * Mxy
    trace_M = Mxx + Myy

    # Compute R scores with k = 0.05
    k = 0.05
    R = det_M - k * (trace_M ** 2)

    # Thresholding
    threshold = 0.01 * R.max()
    R[R < threshold] = 0

    # Non-maximum suppression
    corner_mask = non_maximum_suppression(R)

    return corner_mask


# main function
if __name__ == '__main__':
    # read the image in data
    # rgb image
    rgb_filename = 'cracker_box.jpg'
    im = cv2.imread(rgb_filename)

    # your implementation of the harris corner detector
    corner_mask = harris_corner(im)

    # opencv harris corner
    img = im.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    opencv_mask = dst > 0.01 * dst.max()

    # visualization for your debugging
    fig = plt.figure()

    # show RGB image
    ax = fig.add_subplot(1, 3, 1)
    plt.imshow(im[:, :, (2, 1, 0)])
    ax.set_title('RGB image')

    # show our corner image
    ax = fig.add_subplot(1, 3, 2)
    plt.imshow(im[:, :, (2, 1, 0)])
    index = np.where(corner_mask > 0)
    plt.scatter(x=index[1], y=index[0], c='y', s=5,)
    ax.set_title('our corner image')

    # show opencv corner image
    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(im[:, :, (2, 1, 0)])
    index = np.where(opencv_mask > 0)
    plt.scatter(x=index[1], y=index[0], c='y', s=5,)
    ax.set_title('opencv corner image')

    plt.show()
