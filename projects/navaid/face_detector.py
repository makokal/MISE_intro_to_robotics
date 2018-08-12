
import cv2 as cv
import numpy as np


class FaceDetector(object):

    def __init__(self):
        self._face_cascade = cv.CascadeClassifier('haar_model_frontface.xml')

    def find_faces_haar(self, img):
        """ Find all the regions in the image that have faces.
        
        Parameters
        ----------
        img : The input image.

        Returns
        -------
        The final image, and the list of faces detected.
        """
        # Convert the image to grayscale (black and white) for detection.
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find all the faces. Calls the OpenCV detection API.
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(120, 120),
            maxSize=(300, 300),
            flags=cv.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        return img, faces

    def anonymize_random_colors(self, img, faces):
        """ Anonymize faces by covering face region using randon colors
        
        Parameters
        ----------
        img : The input image.
        faces : A list of faces. Each face is a rectangle with (x, w, w, h)
            Example: [(3,6,9,10), (a,b,c,d)....]

        Returns
        -------
        The final image.
        """
        if len(faces) < 1:
            return img

        for f in faces:
            print(f)
            f_x, f_y, f_w, f_h = f[0], f[1], f[2], f[3]
            img[f_y:f_y+f_w, f_x:f_x+f_h, 0] = np.random.randint(255)
            img[f_y:f_y+f_w, f_x:f_x+f_h, 1] = np.random.randint(255)
            img[f_y:f_y+f_w, f_x:f_x+f_h, 2] = np.random.randint(255)
        return img

    def anonymize_lines(self, img, faces, direction='vertical'):
        """ Anonymize faces by drawing lines (vertical, horizontal, both)
        
        Parameters
        ----------
        img : The input image.
        faces : A list of faces. Each face is a rectangle with (x, w, w, h)

        Returns
        -------
        The final image.
        """
        if len(faces) < 1:
            return img

        for f in faces:
            if direction == 'vertical':
                img = _draw_vertial_lines(img, f)
            elif direction == 'horizontal':
                img = _draw_horizonal_lines(img, f)
            elif direction == 'both':
                img = _draw_vertial_lines(img, f)
                img = _draw_horizonal_lines(img, f)
            else:
                print('Invalid choice of direction.')

        return img

    def pixelate_faces(self, img, faces):
        """ Pixelate the face regions
        
        Parameters
        ----------
        img : The input image.
        faces : A list of faces. Each face is a rectangle with (x, w, w, h)

        Returns
        -------
        The final image with all faces pixelated.
        """
        if len(faces) < 1:
            return img

        for f in faces:
            rects = _make_smaller_rectangles(f, 20, 20)
            for r in rects:
                f_x, f_y, f_w, f_h = r[0], r[1], r[2], r[3]
                
                img[f_y:f_y+f_w, f_x:f_x+f_h, 0] = np.average(img[f_y:f_y+f_w, f_x:f_x+f_h, 0])
                img[f_y:f_y+f_w, f_x:f_x+f_h, 1] = np.average(img[f_y:f_y+f_w, f_x:f_x+f_h, 1])
                img[f_y:f_y+f_w, f_x:f_x+f_h, 2] = np.average(img[f_y:f_y+f_w, f_x:f_x+f_h, 2])

        return img


def _make_smaller_rectangles(f, dx=10, dy=10):
    f_x, f_y, f_w, f_h = f[0], f[1], f[2], f[3]
    rectangles = []
    for step_h in range(f_x, f_x+f_h, dx):
        for step_w in range(f_y, f_y+f_w, dy):
            rectangles.append((step_h, step_w, dx, dy))
    return rectangles


def _draw_vertial_lines(img, face):
    f_x, f_y, f_w, f_h = face[0], face[1], face[2], face[3]
    for step in range(0, f_h, 20):
        img[f_y:f_y+f_w, f_x+step:f_x+step+3, 0] = np.random.randint(255)
        img[f_y:f_y+f_w, f_x+step:f_x+step+3, 1] = np.random.randint(255)
        img[f_y:f_y+f_w, f_x+step:f_x+step+3, 2] = np.random.randint(255)
    return img


def _draw_horizonal_lines(img, face):
    f_x, f_y, f_w, f_h = face[0], face[1], face[2], face[3]
    for step in range(0, f_w, 20):
        img[f_y+step:f_y+step+3, f_x:f_x+f_h, 0] = np.random.randint(255)
        img[f_y+step:f_y+step+3, f_x:f_x+f_h, 1] = np.random.randint(255)
        img[f_y+step:f_y+step+3, f_x:f_x+f_h, 2] = np.random.randint(255)
    return img
