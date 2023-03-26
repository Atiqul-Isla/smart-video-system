# Flask modules
from flask import Flask, render_template, Response, request, flash, url_for, redirect, get_flashed_messages, session
from flask_socketio import SocketIO, emit

# OpenCV
import cv2
import numpy as np

# Basic Python modules
from PIL import Image
import os
import sys

# SSH Connection
import paramiko

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/backend')
yml_dir = os.path.join(BASE_DIR, "backend/trainer.yml")
pickle_dir = os.path.join(BASE_DIR, "backend/labels.pickle")
images_dir = os.path.join(BASE_DIR, "backend/images")

# Hand-made functions
from backend import generate
from backend import Live_Feed
from backend import capture
from backend import trainer

# Hand-made functions in same directory
from SSH_left import connectLeft
from SSH_right import connectRight

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)



@app.route('/', methods=["POST", "GET"])
def index():
    messages = get_flashed_messages()
    return render_template('home.html', messages=messages)

@app.route('/video')
def video():
    return Response(generate.generate_frames(0, images_dir), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/moving')
def moving():
    return Response(generate.generate_frames(1, images_dir), mimetype='multipart/x-mixed-replace; boundary=frame')

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
    if request.method == "POST":
        trainer.trainer(yml_dir, pickle_dir)
        flash('Training completed successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('train.html')

@app.route('/settings', methods=["POST", "GET"])
def settings():
    if request.method == "POST":
        # retrieve the form data submitted by the user
        hst = request.form['hst']
        usr = request.form['usr']
        pwd = request.form['pwd']
        # store the values in the session
        session['hst'] = hst
        session['usr'] = usr
        session['pwd'] = pwd
        return redirect(url_for('index'))
    return render_template('settings.html')

@app.route('/left', methods=["POST", "GET"])
def left():
    if request.method =="POST":
        print("LEFT REQUEST IS SUCCESFUL")
        # os.system("python SSH_left.py")
        # retrieve the parameter values from session
        hst = session.get('hst', '10.0.0.83')
        usr = session.get('usr', 'gurkaran')
        pwd = session.get('pwd', 'gurkaran')
        # call connectLeft with the retrieved values
        connectLeft(hst, usr, pwd)
        return ('', 204)
    
@app.route('/right', methods=["POST", "GET"])
def right():
    if request.method =="POST":
        print("RIGHT REQUEST IS SUCCESFUL")
        # os.system("python SSH_right.py")
        # retrieve the parameter values from session
        hst = session.get('hst', '10.0.0.83')
        usr = session.get('usr', 'gurkaran')
        pwd = session.get('pwd', 'gurkaran')
        # call connectRight with the retrieved values
        connectRight(hst, usr, pwd)
        connectRight("10.0.0.83", "gurkaran", "gurkaran")  
        return ('', 204)

if __name__=="__main__":
    app.run(debug=True)

