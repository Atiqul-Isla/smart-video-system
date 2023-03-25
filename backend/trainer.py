import cv2
import os
from PIL import Image
import numpy as np
import pickle

def trainer(dir, pickle_dir):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Set the path of the directory containing all the subdirectories
    directory = os.path.join(BASE_DIR, "images")
    

    # Create face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Initialize face data and labels lists
    x_faces = []
    labels = []
    current_id = 0
    label_ids = {}
    # Loop through all the subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
                #labels.append(label)
                #faces.append(path)
                print(label, path)
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                print(label_ids)
                pil_image = Image.open(path)
                image_array = np.array(pil_image, "uint8")
                print(image_array)
                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                for (x,y,w,h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_faces.append(roi)
                    labels.append(id_)

    #print(labels)
    #print(faces)

    with open(str(pickle_dir), "wb") as f:
        pickle.dump(label_ids, f)

    if len(labels) > 0:
        recognizer.train(x_faces, np.array(labels))
        recognizer.save(str(dir))
    else:
        recognizer.clear()
        # print("You are in the right spot!")
        if os.path.isfile(str(dir)):
            os.remove(str(dir))
        recognizer.save(str(dir))


