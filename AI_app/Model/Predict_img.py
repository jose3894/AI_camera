# USAGE
# python predict.py --input terrific_natural_disasters_compilation.mp4 --output output/natural_disasters.avi

# import the necessary packages
from tensorflow.keras.models import load_model
from pyimagesearch import config
from collections import deque
import numpy as np
import argparse
from os import walk, remove
from os.path import join
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default=config.INPUT_PATH,
                help="path to our input video")
ap.add_argument("-o", "--output", default=config.OUTPUT_PATH,
                help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128,
                help="size of queue for averaging")
ap.add_argument("-d", "--display", type=int, default=-1,
                help="whether or not output frame should be displayed to screen")
args = vars(ap.parse_args())

# load the trained model from disk
print("[INFO] loading model and label binarizer...")
model = load_model(config.MODEL_PATH)

# initialize the predictions queue
Q = deque(maxlen=args["size"])

# initialize the video stream, pointer to output video file, and
# frame dimensions
print("[INFO] processing images...")

for path, dirs, files in walk(args["input"]):
    for file in files:
        print('\r' + file, end="")
        img = cv2.imread(join(path, file), cv2.IMREAD_UNCHANGED)
        output = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype("float32")

        # make predictions on the frame and then update the predictions
        # queue
        preds = model.predict(np.expand_dims(img, axis=0))[0]
        Q.append(preds)

        # perform prediction averaging over the current history of
        # previous predictions
        results = np.array(Q).mean(axis=0)
        i = np.argmax(results)
        label = config.CLASSES[i]

        # draw the activity on the output frame
        text = "activity: {}".format(label)
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.25, (0, 255, 0), 5)

        cv2.imwrite(join(args["output"], file), output)