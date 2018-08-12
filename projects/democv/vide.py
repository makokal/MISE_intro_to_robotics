cap = cv.VideoCapture(0)
  while True:
      ret, img = cap.read()

      # Detect the faces
      img = find_faces_haar(img, face_cascade)

      # show the result.
      cv.imshow('capture', img)

      # Wait for ESC key, then quit.
      ch = cv.waitKey(1)
      if ch == 27:
          break
  cv.destroyAllWindows()
