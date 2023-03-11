from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
from PIL import Image
import numpy as np
import pickle
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yml_dir = os.path.join(BASE_DIR, "backend/trainer.yml")
pickle_dir = os.path.join(BASE_DIR, "backend/labels.pickle")


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

## Run this first then, run the second line ------------------------------
# camera=cv2.VideoCapture(-1, cv2.CAP_V4L2)
#camera=cv2.VideoCapture(0)
#camera_2=cv2.VideoCapture(1)



def generate_frames(input):

    last_recognized = ('', 0)
    camera=cv2.VideoCapture(input)

    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(yml_dir))

    labels = {"person_name": 1}
    with open(str(pickle_dir), 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}

    while True:
        # Read camera frame
        success, frame = camera.read()
        if not success:
            break
        else:   
            fCascade = cv2.CascadeClassifier('backend/cascades/data/haarcascade_frontalface_alt2.xml')
            grayscale =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = fCascade.detectMultiScale(grayscale, scaleFactor=1.5, minNeighbors=5)
          
            for (x,y,w,h) in faces:
                face = cv2.resize(grayscale[y:y+h, x:x+w], (200, 200))  # Crop and resize face
                id_, confidence = recognizer.predict(face)  # Recognize face
                if confidence>=45 and confidence <=85:  # Threshold for face recognition confidence
                    print(id_)
                    print(labels[id_])
                    name = labels[id_]
                    if name != last_recognized[0] or time.time() - last_recognized[1] >= 300:
                        # If recognized person is different from last recognized person or
                        # more than 5 minutes have passed since last recognition,
                        # send name to alert function and update last recognized person and time
                        last_recognized = (name, time.time())
                        print(name)
                        encoded_name = name.encode()
                        socketio.emit('sendAlert', {'name': name})
                        # yield b'--frame\r\nContent-Type: text/plain\r\n\r\n' + js_command.encode() + b'\r\n'

                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                    # print("{} face recognized (confidence: {})".format(name, confidence))
                else:
                    cv2.putText(frame, "UNKNOWN", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            ret,buffer=cv2.imencode('.jpg', frame)
            frame=buffer.tobytes()
            
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=["POST", "GET"])
def index():
    return render_template('home.html')

@app.route('/video')
def video():
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/moving')
def moving():
    return Response(generate_frames(1), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)

