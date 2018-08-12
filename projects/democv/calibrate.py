import numpy as np
import cv2

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:8].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cap = cv2.VideoCapture(1)
img_ = None
while True:
    ret, img = cap.read()
    if img is None:
        print('Failed to get image')
        continue
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_ = gray

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (6, 8))

    if ret == True:
        print('Found chessboard.')
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (6, 8), corners2,ret)

    cv2.imshow('img',img)
    ch = cv2.waitKey(1)
    if ch == 27:
        break

cv2.destroyAllWindows()

# Find the calibration parameters.
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_.shape[::-1],None,None)
print(camera_matrix)
