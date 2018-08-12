import sys
import numpy as np
import cv2

from matplotlib import pyplot as plt

from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk


REMAP_INTERPOLATION = cv2.INTER_LINEAR

DEPTH_VISUALIZATION_SCALE = 2048

if len(sys.argv) != 2:
    print("Syntax: {0} CALIBRATION_FILE".format(sys.argv[0]))
    sys.exit(1)

calibration = np.load(sys.argv[1], allow_pickle=False)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])

# CAMERA_WIDTH = 1280
# CAMERA_HEIGHT = 720

# TODO: Use more stable identifiers
left = cv2.VideoCapture(1)
right = cv2.VideoCapture(2)

# Increase the resolution
# left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
# left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
# right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
# right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
# left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
# right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# The distortion in the left and right edges prevents a good calibration, so
# discard the edges
CROP_WIDTH = 960
def cropHorizontal(image):
    return image[:,
            int((CAMERA_WIDTH-CROP_WIDTH)/2):
            int(CROP_WIDTH+(CAMERA_WIDTH-CROP_WIDTH)/2)]

# TODO: Why these values in particular?
# TODO: Try applying brightness/contrast/gamma adjustments to the images
stereoMatcher = cv2.StereoSGBM_create()
# stereoMatcher = cv2.StereoBM_create()
stereoMatcher.setMinDisparity(0)
stereoMatcher.setNumDisparities(256)
stereoMatcher.setBlockSize(8)
# stereoMatcher.setROI1(leftROI)
# stereoMatcher.setROI2(rightROI)
stereoMatcher.setSpeckleRange(3)
stereoMatcher.setSpeckleWindowSize(8)

plt.ion()

# Grab both frames first, then retrieve to minimize latency between cameras
while(True):
    if not left.grab() or not right.grab():
        print("No more frames")
        break

    _, leftFrame = left.retrieve()
    # leftFrame = cropHorizontal(leftFrame)
    leftHeight, leftWidth = leftFrame.shape[:2]
    _, rightFrame = right.retrieve()
    # rightFrame = cropHorizontal(rightFrame)
    rightHeight, rightWidth = rightFrame.shape[:2]

    if (leftWidth, leftHeight) != imageSize:
        print("Left camera has different size than the calibration data")
        break

    if (rightWidth, rightHeight) != imageSize:
        print("Right camera has different size than the calibration data")
        break

    fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, REMAP_INTERPOLATION)
    fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY, REMAP_INTERPOLATION)

    # grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    # grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    
    grayLeft = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)

    depth = stereoMatcher.compute(grayLeft, grayRight)

    cv2.imshow('left', leftFrame)
    cv2.imshow('right', rightFrame)

    selem = disk(6)
    result = dilation(depth, selem)
    plt.imshow(result)
    # plt.colorbar()

    cv2.imshow('depth', depth * DEPTH_VISUALIZATION_SCALE)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

left.release()
right.release()
cv2.destroyAllWindows()