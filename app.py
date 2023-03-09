from flask import Flask, render_template, Response
# from camera import Video
import cv2

app=Flask(__name__)

## Run this first then, run the second line ------------------------------
# camera=cv2.VideoCapture(-1, cv2.CAP_V4L2)
#camera=cv2.VideoCapture(0)
#camera_2=cv2.VideoCapture(1)

def generate_frames(input):

    camera=cv2.VideoCapture(input)
    while True:
        # Read camera frame
        success, frame = camera.read()
        if not success:
            break
        else:   
            fCascade = cv2.CascadeClassifier('backend/cascades/data/haarcascade_frontalface_alt2.xml')
            pCascade = cv2.CascadeClassifier('backend/cascades/data/haarcascade_profileface.xml')
            grayscale =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = fCascade.detectMultiScale(grayscale, scaleFactor=1.5, minNeighbors=5)
            r_side_faces = pCascade.detectMultiScale(grayscale, scaleFactor=1.5, minNeighbors=5)

            flipped = cv2.flip(grayscale, 1)
            l_side_faces = pCascade.detectMultiScale(flipped, scaleFactor=1.5, minNeighbors=5)
        

            for (x,y,w,h) in faces:
                print(x,y,w,h)
                grayROI = grayscale[y:y+h,x:x+w]
                colorROI = frame[y:y+h,x:x+w]
                
                color = (255,0,0) #standard blue color
                stroke = 2
                width = x + w
                height = y + h
                cv2.rectangle(frame, (x,y), (width, height), color, stroke)
            
            for (x,y,w,h) in r_side_faces:
                print(x,y,w,h)
                grayROI = grayscale[y:y+h,x:x+w]
                colorROI = frame[y:y+h,x:x+w]
                
                color = (255,0,0) #standard blue color
                stroke = 2
                width = x + w
                height = y + h
                cv2.rectangle(frame, (x,y), (width, height), color, stroke)
            
            for (x1,y,w,h) in l_side_faces:
                print(x,y,w,h)
                grayROI = flipped[y:y+h,x1:x1+w]
                colorROI = frame[y:y+h,x1:x1+w]
                img_item = "my-image.png"
                cv2.imwrite(img_item,grayROI)
                color = (255,0,0) #standard blue color
                stroke = 2
                height = y + h
                cv2.rectangle(frame, (x,y), (width, height), color, stroke)

            ret,buffer=cv2.imencode('.jpg', frame)
            frame=buffer.tobytes()
            
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
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

