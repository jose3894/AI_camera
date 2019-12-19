from flask import Flask, render_template, Response
from camera import CameraStream
import cv2

# START YOLO
import argparse
from os import path, system
import time
import logging
import sys
import numpy as np
import cv2
#from picamera.array import PiRGBArray
#from picamera import PiCamera

from object_detector_detection_api import ObjectDetectorDetectionAPI
from yolo_darfklow import YOLODarkflowDetector
from object_detector_detection_api_lite import ObjectDetectorLite
from utils.utils import Models
from detection_stream import DetectionStream

basepath = path.dirname(__file__)
print(basepath)

# initiate the parser
parser = argparse.ArgumentParser(prog='test_models.py')

# add arguments
parser.add_argument("--model_name", "-mn", type=Models.from_string,
                    required=True, choices=list(Models),
                    help="name of detection model: {}".format(list(Models)))
parser.add_argument("--graph_path", "-gp", type=str, required=False,
                    default=path.join(basepath, "frozen_inference_graph.pb"),
                    help="path to ssdlight model frozen graph *.pb file")
parser.add_argument("--cfg_path", "-cfg", type=str, required=False,
                    default=path.join(basepath, "tiny-yolo-voc.cfg"),
                    help="path to yolo *.cfg file")
parser.add_argument("--weights_path", "-w", type=str, required=False,
                    default=path.join(basepath, "tiny-yolo-voc.weights"),
                    help="path to yolo weights *.weights file")

# read arguments from the command line
args = parser.parse_args()

print('Model loading...')
graph_path = r'/app/Production/frozen_inference_graph.pb'
cfg_path = r'/app/Production/tiny-yolo-voc.cfg'
weights_path = r'/app/Production/tiny-yolo-voc.weights'
#if args.model_name == Models.ssd_lite:
#    predictor = ObjectDetectorDetectionAPI(graph_path)
#elif args.model_name == Models.tiny_yolo:
predictor = YOLODarkflowDetector(cfg_path, weights_path)
#elif args.model_name == Models.tf_lite:
#    predictor = ObjectDetectorLite()

count_img = 0
# END YOLO

app = Flask(__name__)

cap = CameraStream().start()


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen_frame():
    global count_img
    """Video streaming generator function."""
    while cap:
        frame = cap.read()
        system("echo readdd")
        frame = DetectionStream().detect(frame, predictor, count_img)
        if count_img < 31:
            count_img += 1

        convert = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + convert + b'\r\n') # concate frame one by one and show result



@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)