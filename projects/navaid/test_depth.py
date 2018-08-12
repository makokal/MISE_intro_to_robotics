
import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt

from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk


F = 1.42351297e+03
T = 0.15


def fraction_greater(d, value=2.0):
    return sum(d[d < value]) / float(d.size)


def test_depth(left, right):
    stereoMatcher = cv.StereoSGBM_create()
    # stereoMatcher = cv2.StereoBM_create()
    stereoMatcher.setMinDisparity(0)
    stereoMatcher.setNumDisparities(256)
    stereoMatcher.setBlockSize(8)
    # stereoMatcher.setROI1(leftROI)
    # stereoMatcher.setROI2(rightROI)
    stereoMatcher.setSpeckleRange(3)
    stereoMatcher.setSpeckleWindowSize(8)


    # left = cv.blur(left, (5,5))
    # right = cv.blur(right, (5,5))

    grayLeft = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
    grayRight = cv.cvtColor(right, cv.COLOR_BGR2GRAY)

    depth = stereoMatcher.compute(grayLeft, grayRight)

    depth = depth[:, 300:]

    selem = disk(8)
    result = dilation(depth, selem)
    result = erosion(result, selem)
    plt.imshow(result, cmap='viridis')

    distances = np.ones_like(result) * 10000.0
    non_zero = result > 0.0
    distances[non_zero] = (F * T) / result[non_zero]

    print(fraction_greater(distances, 1.0))

    plt.imshow(result)
    plt.colorbar()

    # plt.figure()
    # plt.plot(distances[distances < 10000])
    # plt.colorbar()

    plt.show()


if __name__ == '__main__':
    left = cv.imread('../democv/Couch-perfect/im0.png')
    right = cv.imread('../democv/Couch-perfect/im1.png')

    test_depth(left, right)