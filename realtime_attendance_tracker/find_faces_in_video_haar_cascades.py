import cv2
import sys
from time import time

# Average FPS : 0.03842847405410395 in frames: 205

cascPath = './haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

t = 0.0
n = 0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    start = time()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    t = t + time() - start
    n = n + 1

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Average time per frame : " + str(t/n) + " in frames: " + str(n))

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()