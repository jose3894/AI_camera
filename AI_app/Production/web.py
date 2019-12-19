from flask import Flask, render_template, Response
from camera import CameraStream
import cv2
from os.path import join
from tensorflow.lite.python.interpreter import Interpreter
import numpy as np

app = Flask(__name__)

cap = CameraStream().start()

MODEL_NAME = r'/app/Production/Sample_TF_model'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
min_conf_threshold = 0.5
resolution = '640x480'
resW, resH = resolution.split('x')
imW, imH = int(resW), int(resH)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = join(MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = join(MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen_frame():
    """Video streaming generator function."""
    while cap:
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        frame1 = cap.read()
        frame2 = frame1



        convert = cv2.imencode('.jpg', frame2)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + convert + b'\r\n') # concate frame one by one and show result

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)