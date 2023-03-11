import numpy as np
import cv2
import pickle

fCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture every frame
    ret, frame = cap.read()

    #Turning frame into gray scale because according to documentation thats the only color that is usable
    grayscale =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = fCascade.detectMultiScale(grayscale, scaleFactor=1.5, minNeighbors=5)
    # r_side_faces = pCascade.detectMultiScale(grayscale, scaleFactor=1.5, minNeighbors=5)

    # flipped = cv2.flip(grayscale, 1)
    # l_side_faces = pCascade.detectMultiScale(flipped, scaleFactor=1.5, minNeighbors=5)
 

    for (x,y,w,h) in faces:
        # Recognize face
        face = cv2.resize(grayscale[y:y+h, x:x+w], (200, 200))  # Crop and resize face
        id_, confidence = recognizer.predict(face)  # Recognize face
        if confidence>=45 and confidence <=85:  # Threshold for face recognition confidence
            print(id_)
            print(labels[id_])
            name = labels[id_]
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            # print("{} face recognized (confidence: {})".format(name, confidence))
        else:
            cv2.putText(frame, "UNKNOWN", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
      
    # Display each color frame (NOT gray frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()