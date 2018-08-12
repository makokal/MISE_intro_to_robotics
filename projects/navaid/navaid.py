
import argparse
import os
import os.path

import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt

from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk

from face_detector import *
from speak_directions import *

from stereovision.stereo_cameras import StereoPair
from progressbar import ProgressBar, Bar, Percentage
from stereovision.stereo_cameras import ChessboardFinder, CalibratedPair
from stereovision.blockmatchers import StereoBM, StereoSGBM
from stereovision.calibration import StereoCalibration
from stereovision.ui_utils import calibrate_folder, CHESSBOARD_ARGUMENTS
from stereovision.ui_utils import find_files


T = 0.8
F = 1.42351297e+03


class NavAid(object):
    def __init__(self):
        # Setup face detector.
        self._detector = FaceDetector()
        self._speaker = Speaker()

        print('='*80 + '\nSetting up camera feed...')

        self._cameras = StereoPair([1, 2])

        print('-'*80 + '\nNavAid property initialized. Ready to move!')

        self.stereoMatcher = cv.StereoSGBM_create()
        # self.stereoMatcher = cv2.StereoBM_create()
        self.stereoMatcher.setMinDisparity(0)
        self.stereoMatcher.setNumDisparities(256)
        self.stereoMatcher.setBlockSize(8)
        # self.stereoMatcher.setROI1(leftROI)
        # self.stereoMatcher.setROI2(rightROI)
        self.stereoMatcher.setSpeckleRange(3)
        self.stereoMatcher.setSpeckleWindowSize(8)
    
    def run(self, full_operation_mode=False):
        if full_operation_mode:
            self.setup_full_operation()
        
        plt.ion()
    
        while True:
            left_image, right_image = self._cameras.get_frames()
            if left_image is not None and right_image is not None:
                left_imgage, right_image = self.anonymize(left_image, right_image)
                cv.imshow('Left', left_image)
                cv.imshow('Right', right_image)

                if full_operation_mode:
                    distances = self.process_depth_view(left_image, right_image)
                    self.send_instructions(distances)

                self._speaker._engine.runAndWait()
                ch = cv.waitKey(1)
                if ch == 27:
                    break
        cv.destroyAllWindows()
    
    def setup_full_operation(self):
        print('WARNING: This step should be run after calibration is successful.')
        self._stereo_rig = CalibratedPair(
            None,
            StereoCalibration(input_folder='calib'),
            StereoBM())
    
    def process_depth_view2(self, left, right):       
        # Rectify images.
        rectified_pair = self._stereo_rig.calibration.rectify([left, right])

        cv.imshow('rect_left', rectified_pair[0])
        cv.imshow('rect_right', rectified_pair[1])

        points, disparity = self._stereo_rig.get_point_cloud(rectified_pair)
        points = points.filter_infinity()
        # points.write_ply('output/pcd.ply')

        # DEPTH_VISUALIZATION_SCALE = 2048
        # cv.imshow('depth', disparity / DEPTH_VISUALIZATION_SCALE)
        cv.imshow('depth', disparity)
        plt.imshow(disparity)
        # plt.show()
    
    def process_depth_view(self, left, right):
        grayLeft = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
        grayRight = cv.cvtColor(right, cv.COLOR_BGR2GRAY)

        depth = self.stereoMatcher.compute(grayLeft, grayRight)

        selem = disk(4)
        result = dilation(depth, selem)
        result = erosion(result, selem)
        plt.imshow(result, cmap='viridis')

        distances = np.ones_like(result) * 10000.0
        non_zero = result > 0.0
        distances[non_zero] = (F * T) / result[non_zero]

        # for i in range(result.shape[0]):
        #     for j in range(result.shape[1]):
        #         if result[i, j] > 0.0:
        #             distances[i, j] = (F * T) / result[i, j]
        #         else:
        #             result[i, j] = 10000.0
        return distances

    def send_instructions(self, depth_view):
        # Convert depth to instruction
        print('Distances', depth_view)

        if (depth_view < 2.0).count():
            instruction = WARNING + PAUSE + ' obstacle ' + AHEAD
            self._speaker.say_direction(instruction)
        else:
            self._speaker.say_direction('All clear ahead.')
        
        return True
        
    def anonymize(self, left, right):
        left, left_faces = self._detector.find_faces_haar(left)
        left = self._detector.pixelate_faces(left, left_faces)

        right, right_faces = self._detector.find_faces_haar(right)
        right = self._detector.pixelate_faces(right, right_faces)
        
        return left, right
        
    
    def prepare_to_calibrate(self, num_pictures, output_folder='output'):
        progress = ProgressBar(maxval=num_pictures, widgets=[Bar("=", "[", "]"), " ", Percentage()])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        progress.start()
        with ChessboardFinder((1, 2)) as pair:
            for i in range(num_pictures):
                frames = pair.get_chessboard(9, 7, True)
                for side, frame in zip(("left", "right"), frames):
                    number_string = str(i + 1).zfill(len(str(num_pictures)))
                    filename = "{}_{}.png".format(side, number_string)
                    output_path = os.path.join(output_folder, filename)
                    cv.imwrite(output_path, frame)
                progress.update(progress.maxval - (num_pictures - i))
                for i in range(10):
                    pair.show_frames(1)
            progress.finish()
    
    def calibrate(self, output_folder='output', calibration_folder='calib'):
        # Make the calibration step.
        print('>' * 5 + ' Performing calibration....')

        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.rows = 7
        args.columns = 9
        args.square_size = 2.2
        args.input_files = find_files(output_folder)
        args.output_folder = output_folder
        args.calibrate_folder = calibrate_folder
        args.show_chessboards = True
        calibrate_folder(args)