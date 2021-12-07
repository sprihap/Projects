from PIL import Image
import face_recognition
import cv2
import matplotlib.pyplot as plt

imname = "friends_banner.jpeg"

# Load the jpg file into a numpy array
image = face_recognition.load_image_file(imname)

# Find all the faces in the image using convolutional neural network.
# This method is more accurate than the default HOG model, but it's slower
# unless you have an nvidia GPU and dlib compiled with CUDA extensions.
face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0,\
     model="cnn")

print("Found {} face(s) in this photograph.".format(len(face_locations)))
image = cv2.imread(imname)
for (i, face_location) in enumerate(face_locations):

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}"\
        .format(top, left, bottom, right))

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    # pil_image = Image.fromarray(face_image)
    # pil_image.show()
    plt.imshow(face_image)

    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    # show the face number
    cv2.putText(image, "Face #{}".format(i + 1), (left - 10, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


cv2.imwrite('cnn_'+imname, image)