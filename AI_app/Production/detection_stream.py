import argparse
from os import path
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



basepath = path.dirname(__file__)

class DetectionStream:

    def detect(self, frame, predictor):
        image = frame
        result = predictor.detect(image)

        for obj in result:

            cv2.rectangle(image, obj[0], obj[1], (0, 255, 0), 2)

        return image


