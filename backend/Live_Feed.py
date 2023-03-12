import cv2

def generateFeed():
    # Create face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Set up video capture device
    cap = cv2.VideoCapture(0)
    
    # Loop through frames of video stream
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Display image with face detection overlay
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Check if face is in a good position for capturing image
            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                if w > 200 and h > 200 and x > 50 and y > 50 and x+w < 590 and y+h < 430:
                    cv2.putText(frame, "Face in good position!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret,buffer=cv2.imencode('.jpg', frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
   