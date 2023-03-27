import cv2
import pickle 
import os
import time
import smtplib
import csv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def generate_frames(input, dir, video_dir, email, email_pwd):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    yml_dir = os.path.join(BASE_DIR, "trainer.yml")
    pickle_dir = os.path.join(BASE_DIR, "labels.pickle")
    image_dir = os.path.join(BASE_DIR, "/backend/images")
    # video_dir = os.path.join(BASE_DIR, "/backend/videos")

    last_recognized = ('', 0.0)
                
    camera=cv2.VideoCapture(input)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(yml_dir))

    labels = {"person_name": 1}
    with open(str(pickle_dir), 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}

    record = False  # Flag to indicate whether to record or not
    start_time = None  # Time when recording started
    last_seen_time = None  # Time when face was last seen
    out = None  # Video writer object

    # Email setup
    email_address = email  # Replace with your email address
    email_password = email_pwd  # Replace with your email password
    recipient_email = email  # Replace with recipient email address

    def send_email(image_path, person_name, body, subject):
        message = MIMEMultipart()
        message['From'] = email_address
        message['To'] = recipient_email
        message['Subject'] = subject

        msg_body = body
        message.attach(MIMEText(msg_body, "plain"))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(email_address, email_password)
            smtp.send_message(message)

    while True:
        # Read camera frame
        success, frame = camera.read()
        if not success:
            break
        else:   
            fCascade = cv2.CascadeClassifier('backend/cascades/data/haarcascade_frontalface_alt2.xml')
            grayscale =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = fCascade.detectMultiScale(grayscale, scaleFactor=1.5, minNeighbors=5)

            # Display the number of faces detected
            cv2.putText(frame, f"Faces Detected: {len(faces)}", (50, frame.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        
            for (x,y,w,h) in faces:
                if os.listdir(dir):
                    face = cv2.resize(grayscale[y:y+h, x:x+w], (200, 200))  # Crop and resize face
                    id_, confidence = recognizer.predict(face)  # Recognize face
                    if not record:
                        # Start recording if a face is detected
                        record = True
                        start_time = int(time.time())
                        out = cv2.VideoWriter(f'{video_dir}/{id_}_{start_time}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame.shape[1], frame.shape[0]))
                        print("Video Finished Capturing!!")
                        print(f'{video_dir}/{id_}.mp4')
                    if confidence>=45 and confidence <=85:  # Threshold for face recognition confidence
                        name = labels[id_]
                        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

                        if name != last_recognized[0] and time.time() - last_recognized[1] >= 300:
                        # Send email notification if the recognized name is different from the last recognized name, or if more than 5 minutes have passed since the last notification
                            if name != 'UNKNOWN':
                                subject = f"{name} detected"
                                body = f"{name} was detected at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
                            send_email(image_dir, name, body, subject)  # Replace this with your function to send an email notification
                            last_recognized = (name, time.time())
                    
                    last_seen_time = time.time()
                else:
                    # print('You should be here !!')
                    cv2.putText(frame, "UNKNOWN", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                    if time.time() - last_recognized[1] >= 300:
                        subject = "Unknown person detected"
                        body = f"An unknown person was detected at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
                        send_email(image_dir, name, body, subject)  # Replace this with your function to send an email notification
                        last_recognized = (name, time.time())
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                
                
            # Check if face has been undetectable for at least 5 seconds
            if record and last_seen_time is not None and time.time() - last_seen_time >= 5:
                # Stop recording
                record = False
                out.release()
                
            # Write frame to video if recording
            if record:
                out.write(frame)
                
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            ret,buffer=cv2.imencode('.jpg', frame)
            frame=buffer.tobytes()
        
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    