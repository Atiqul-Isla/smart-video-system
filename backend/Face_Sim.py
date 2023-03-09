# import numpy as np
# import cv2

# fCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
# pCascade = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')
# cap = cv2.VideoCapture(0)

# while(True):
#     # Capture every frame
#     ret, frame = cap.read()

#     #Turning frame into gray scale because according to documentation thats the only color that is usable
#     grayscale =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = fCascade.detectMultiScale(grayscale, scaleFactor=1.5, minNeighbors=5)
#     r_side_faces = pCascade.detectMultiScale(grayscale, scaleFactor=1.5, minNeighbors=5)

#     flipped = cv2.flip(grayscale, 1)
#     l_side_faces = pCascade.detectMultiScale(flipped, scaleFactor=1.5, minNeighbors=5)
 

#     for (x,y,w,h) in faces:
#         print(x,y,w,h)
#         grayROI = grayscale[y:y+h,x:x+w]
#         colorROI = frame[y:y+h,x:x+w]
        
#         color = (255,0,0) #standard blue color
#         stroke = 2
#         width = x + w
#         height = y + h
#         cv2.rectangle(frame, (x,y), (width, height), color, stroke)
    
#     for (x,y,w,h) in r_side_faces:
#         print(x,y,w,h)
#         grayROI = grayscale[y:y+h,x:x+w]
#         colorROI = frame[y:y+h,x:x+w]
        
#         color = (255,0,0) #standard blue color
#         stroke = 2
#         width = x + w
#         height = y + h
#         cv2.rectangle(frame, (x,y), (width, height), color, stroke)
    
#     for (x1,y,w,h) in l_side_faces:
#         print(x,y,w,h)
#         grayROI = flipped[y:y+h,x1:x1+w]
#         colorROI = frame[y:y+h,x1:x1+w]
#         img_item = "my-image.png"
#         cv2.imwrite(img_item,grayROI)
#         color = (255,0,0) #standard blue color
#         stroke = 2
#         height = y + h
#         cv2.rectangle(frame, (x,y), (width, height), color, stroke)
    

#     # Display each color frame (NOT gray frame)
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break


# cap.release()
# cv2.destroyAllWindows()