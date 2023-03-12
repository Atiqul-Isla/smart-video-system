# Flask modules
from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, emit

# OpenCV
import cv2
import numpy as np

# Basic Python modules
from PIL import Image
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/backend')
yml_dir = os.path.join(BASE_DIR, "backend/trainer.yml")
pickle_dir = os.path.join(BASE_DIR, "backend/labels.pickle")

# Hand-made functions
from backend import generate
from backend import Live_Feed
from backend import capture

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)



@app.route('/', methods=["POST", "GET"])
def index():

    return render_template('home.html')

@app.route('/video')
def video():
    return Response(generate.generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/moving')
def moving():
    return Response(generate.generate_frames(1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add', methods=["POST", "GET"])
def add():
    if request.method == "POST":
        name = request.form['name']
        n=10
        capture.capture(name, n)
        return render_template('train.html', name=name)
    else:
        return render_template('add.html')

@app.route('/live')
def live():
    return Response(Live_Feed.generateFeed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train', methods=["POST", "GET"])
def train():

    return render_template('train.html')

# @socketio.on("connect")
# def connect(auth):
#     print("Client connected")
#     socketio.emit("connected", {"data": "Connected"})

if __name__=="__main__":
    app.run(debug=True)

