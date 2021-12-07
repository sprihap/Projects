import cv2
import sys
from time import time
import face_recognition

# Average FPS : 1.8839163621266684 in frames: 60 - CNN

video_capture = cv2.VideoCapture(0)

t = 0.0
n = 0
while True:
    # Capture frame-by-frame
    ret, image = video_capture.read()

    start = time()


    # Find all the faces in the image using the default HOG-based model that is provided by
    # dlib and face recognition library on top of it.
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, \
        model="cnn")

    t = t + time() - start
    n = n + 1


    print("I found {} face(s) in this photograph.".format(len(face_locations)))
    for (i, face_location) in enumerate(face_locations):

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}"\
            .format(top, left, bottom, right))

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (left - 10, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Display the resulting frame
    cv2.imshow('Video', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Average time per frame : " + str(t/n) + " in frames: " + str(n))

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()