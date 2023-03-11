import cv2
import os

# Create face detector and recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.EigenFaceRecognizer_create()

# Set up video capture device
cap = cv2.VideoCapture(0)

# Prompt user to enter their name
name = input("Enter your name: ")

# Create directory for user's face images
user_dir = os.path.join("images/", name)
if not os.path.exists(user_dir):
    os.makedirs(user_dir)

# Capture 10 face images
for i in range(10):
    # Capture frame from video stream
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Display image with face detection overlay
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display captured frame
    cv2.imshow("Capture", frame)

    # Save face image if only one face is detected
    if len(faces) == 1:
        face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))  # Crop and resize face
        filename = os.path.join(user_dir, "{}.jpg".format(i))
        cv2.imwrite(filename, face)
        print("Saved face image: {}".format(filename))

    # Wait for user to press 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture device and close windows
cap.release()
cv2.destroyAllWindows()
