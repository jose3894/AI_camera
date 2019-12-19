import argparse
from os import path
import time
import logging
import sys
import numpy as np
import cv2
from os import makedirs
from os.path import exists
#from picamera.array import PiRGBArray
#from picamera import PiCamera

from object_detector_detection_api import ObjectDetectorDetectionAPI
from yolo_darfklow import YOLODarkflowDetector
from object_detector_detection_api_lite import ObjectDetectorLite
from utils.utils import Models


class DetectionStream:

    def detect(self, frame, predictor, count_img):

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dim = (416, 416)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        h, w, _ = image.shape

        result = predictor.detect(image)

        for obj in result:
            cv2.rectangle(image, obj[0], obj[1], (0, 255, 0), 2)
            cv2.putText(image, '{}: {:.2f}'.format(obj[3], obj[2]),
                        (obj[0][0], obj[0][1] - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        dim = (640, 480)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Make directory output
        if not exists(r'/output'):
            makedirs(r'/output', exist_ok=True)

        if count_img < 30:
            cv2.imwrite(r'/output/img' + str(count_img) + '.jpg', frame)

        return frame


