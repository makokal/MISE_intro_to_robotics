
import sys

import cv2 as cv


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness=1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


def find_faces_haar(frame, face_cascade):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame


def find_faces_hog(img, hog):
    found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
        else:
            found_filtered.append(r)
    draw_detections(img, found)
    draw_detections(img, found_filtered, 3)
    return img


if __name__ == '__main__':
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    face_cascade = cv.CascadeClassifier(sys.argv[1])

    cap = cv.VideoCapture(1)
    while True:
        ret, img = cap.read()

        # face detection using HOG
        # img = find_faces_hog(img, hog=hog)

        # HAAR
        img = find_faces_haar(img, face_cascade)

        # show the result.
        cv.imshow('capture', img)
        ch = cv.waitKey(1)
        if ch == 27:
            break
    cv.destroyAllWindows()
