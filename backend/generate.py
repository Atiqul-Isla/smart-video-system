import cv2
import socketio
import pickle 
import os
import time



def generate_frames(input):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    yml_dir = os.path.join(BASE_DIR, "trainer.yml")
    pickle_dir = os.path.join(BASE_DIR, "labels.pickle")

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

            # Display the number of faces detected
            cv2.putText(frame, f"Faces Detected: {len(faces)}", (50, frame.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
          
            for (x,y,w,h) in faces:
                face = cv2.resize(grayscale[y:y+h, x:x+w], (200, 200))  # Crop and resize face
                id_, confidence = recognizer.predict(face)  # Recognize face
                if confidence>=45 and confidence <=85:  # Threshold for face recognition confidence
                    print(id_)
                    print(labels[id_])
                    name = labels[id_]
    

                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                    print("{} face recognized (confidence: {})".format(name, confidence))
                else:
                    cv2.putText(frame, "UNKNOWN", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            ret,buffer=cv2.imencode('.jpg', frame)
            frame=buffer.tobytes()
            
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')