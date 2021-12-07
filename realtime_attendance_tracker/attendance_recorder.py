import face_recognition
import cv2
import numpy as np
from datetime import datetime

video_capture = cv2.VideoCapture(0)
DISTANCE_THRESHOLD = 0.55
PRESENCE_THRESHOLD = 0

COMPRESS_RATIO = 0.25

# Training on known faces. These are written here for simplicity of understanding
# but when class data is available, we can move this to a separate script.

# Load a sample picture and learn how to recognize it.
spriha_image = face_recognition.load_image_file("train/Spriha/Spa.jpeg")
spriha_face_encoding = face_recognition.face_encodings(spriha_image)[0]

# Load a second sample picture and learn how to recognize it.
sushma_image = face_recognition.load_image_file("train/Sushma/Sushma3.jpeg")
sushma_face_encoding = face_recognition.face_encodings(sushma_image)[0]

# Load a third sample picture and learn how to recognize it.
aaron_image = face_recognition.load_image_file("train/Aaron/Aaron.jpeg")
aaron_face_encoding = face_recognition.face_encodings(aaron_image)[0]

# Load a fourth sample picture and learn how to recognize it.
atharva_image = face_recognition.load_image_file("train/Atharva/Atharva.jpeg")
atharva_face_encoding = face_recognition.face_encodings(atharva_image)[0]


# Create arrays of known face encodings and their names. This can be saved to a file
# when training is done in separate script on large scale.

known_face_encodings = [
    spriha_face_encoding,
    sushma_face_encoding,
    aaron_face_encoding,
    atharva_face_encoding
]
known_face_names = [
    "Spriha",
    "Sushma",
    "Aaron",
    "Atharva"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

attendees = {}
current_count = 0
n = 0

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to compress size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=COMPRESS_RATIO, fy=COMPRESS_RATIO)

    # Convert the image from BGR color (which OpenCV uses) to RGB color 
    # (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        n = n + 1

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for name in attendees.keys():
            attendees[name]['present'] = 0

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index] < DISTANCE_THRESHOLD:
                name = known_face_names[best_match_index]
                if name not in attendees:
                    attendees[name] = {'present': 0, 'count' : 0}
                attendees[name]['present'] = 1
                attendees[name]['count'] = attendees[name]['count'] + 1

                

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        if name in attendees:
            attendees[name]['image'] = frame[top:bottom, left:right]

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    if current_count < len(attendees):
        current_count = len(attendees)
        print("Current attendance: " + str(current_count))


    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyWindow('Video')
print(attendees)
print('Total frames processed: ' + str(n))

# Generate a collage of current students attendees for run.
if attendees:
    images = []
    row = []
    count = 0
    for name in attendees:
        if attendees[name]['count'] / n > PRESENCE_THRESHOLD:
            frame = cv2.resize(attendees[name]['image'], (150, 150))

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            if len(row) > 0:
                row = np.hstack([row, frame])
            else:
                row = frame

        count = count + 1
        if count % 10 == 0 or count == len(attendees):
            if len(images) > 0:
                images = np.vstack([images, row])
            else:
                images = row

            row = []

    cv2.imshow('Class attendees', images)
    cv2.waitKey(0)
    now = datetime.now()
    cv2.imwrite("attendance_" + now.strftime("%Y-%m-%d_%H_%M_%S") + ".jpeg", images)

cv2.destroyAllWindows()
