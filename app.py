# Flask modules
from flask import Flask, render_template, Response, request, flash, url_for, redirect, get_flashed_messages, session, send_file,send_from_directory
from flask_socketio import SocketIO, emit

# OpenCV
import cv2
import numpy as np

# Basic Python modules
from PIL import Image
import os
import sys
import csv
import shutil
import zipfile

# SSH Connection
import paramiko

# List of important directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/backend')
yml_dir = os.path.join(BASE_DIR, "backend/trainer.yml")
pickle_dir = os.path.join(BASE_DIR, "backend/labels.pickle")
images_dir = os.path.join(BASE_DIR, "backend/images")
csv_dir = os.path.join(BASE_DIR, "backend/capture_history.csv")
videos_dir = os.path.join(BASE_DIR, "backend/videos")

# Hand-made functions
from backend import generate
from backend import Live_Feed
from backend import capture
from backend import trainer

from hardware import SSH_left
from hardware import SSH_right

# Init Flask Web App and Secret Key to store sessions
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Create Home route
@app.route('/', methods=["POST", "GET"])
def index():
    messages = get_flashed_messages()
    hst = session.get('hst')
    usr = session.get('usr')
    pwd = session.get('pwd')
    email = session.get('email')
    email_pwd = session.get('email_pwd')
    return render_template('home.html', messages=messages, hst=hst, usr=usr, pwd=pwd, email=email, email_pwd=email_pwd)

# Create stationary camera interface
@app.route('/video', methods=["POST", "GET"])
def video():
    email = session.get('email')
    email_pwd = session.get('email_pwd')
    return Response(generate.generate_frames(0, images_dir, videos_dir, email, email_pwd), mimetype='multipart/x-mixed-replace; boundary=frame')

# Create moving camera interface
@app.route('/moving', methods=["POST", "GET"])
def moving():
    email = session.get('email')
    email_pwd = session.get('email_pwd')
    return Response(generate.generate_frames(1, images_dir, videos_dir, email, email_pwd), mimetype='multipart/x-mixed-replace; boundary=frame')

# Create Add route to capture training data
@app.route('/add', methods=["POST", "GET"])
def add():
    if request.method == "POST":
        name = request.form['name']
        n = request.form['num']
        capture.capture(name, n)
        return render_template('train.html', name=name)
    else:
        return render_template('add.html')

# Add live route to display live feed in Add
@app.route('/live')
def live():
    return Response(Live_Feed.generateFeed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Create Train route to train model and store new information in YML file
@app.route('/train', methods=["POST", "GET"])
def train():
    if request.method == "POST":
        trainer.trainer(yml_dir, pickle_dir)
        flash('Training completed successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('train.html')

# Create Settings route to set machine and email parameters
@app.route('/settings', methods=["POST", "GET"])
def settings():
    if request.method == "POST":
        # retrieve the form data submitted by the user
        hst = request.form['hst']
        usr = request.form['usr']
        pwd = request.form['pwd']
        email = request.form['email']
        email_pwd = request.form['email_pwd']
        # store the values in the session
        session['hst'] = hst
        session['usr'] = usr
        session['pwd'] = pwd
        session['email'] = email
        session['email_pwd'] = email_pwd
        return redirect(url_for('index'))
    return render_template('settings.html')

# Create Left route for turning the camera left
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
        SSH_left.connectLeft(hst, usr, pwd)
        return ('', 204)

# Create Right route for turning the camera right
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
        SSH_right.connectRight(hst, usr, pwd)
        return ('', 204)

# Create History route to store history   
@app.route('/history')
def history():
    with open(csv_dir, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
    return render_template('history.html', data=data, video_files=video_files)

# Allow downloading of captured videos
@app.route('/video/<filename>')
def download_video(filename):
    return send_from_directory(videos_dir, filename, as_attachment=True)

# Allow downloads of history, training data and labels
@app.route('/download_history')
def download_history():
    return send_file(csv_dir, as_attachment=True)

@app.route('/download_training')
def download_training():
    return send_file(yml_dir, as_attachment=True)

@app.route('/download_labels')
def download_labels():
    return send_file(pickle_dir, as_attachment=True)

# Download all images/ Training Data in a zip
@app.route('/download_images')
def download_images():
    # Create a temporary directory to store the zipped images
    tmp_dir = os.path.join(BASE_DIR, "tmp")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Zip each user folder in the images directory
    for user_dir in os.listdir(images_dir):
        user_path = os.path.join(images_dir, user_dir)
        if os.path.isdir(user_path):
            zip_path = os.path.join(tmp_dir, "{}.zip".format(user_dir))
            shutil.make_archive(zip_path[:-4], 'zip', user_path)

    # Zip all the user folders together
    zip_path = os.path.join(tmp_dir, "images.zip")
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        for user_dir in os.listdir(images_dir):
            user_path = os.path.join(images_dir, user_dir)
            if os.path.isdir(user_path):
                for file_name in os.listdir(user_path):
                    file_path = os.path.join(user_path, file_name)
                    zip_file.write(file_path, os.path.join(user_dir, file_name))

    # Download the zipped images directory
    return send_file(zip_path, as_attachment=True)

# Call main and run app
if __name__=="__main__":
    app.run(debug=True)

