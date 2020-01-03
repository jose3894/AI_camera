from tensorflow.keras.models import load_model
from pyimagesearch import config
from collections import deque
import numpy as np
from os import walk, makedirs
from shutil import rmtree
from os.path import join, exists, basename
import cv2

# Delete old folder save and make new one
if exists(config.OUTPUT_PATH):
    print("Deleting old folder: " + config.OUTPUT_PATH)
    rmtree(config.OUTPUT_PATH)
makedirs(config.OUTPUT_PATH)

# load the trained model from disk
print("[INFO] loading model and label binarizer...")
model = load_model(config.MODEL_PATH)

# Proccessing images
print("[INFO] processing images...")

total_files = 0
label_counter = 0
label_found = False

with open(config.LOG_ACCURACY, 'w') as accuracy_file:
    for path, dirs, files in walk(config.INPUT_PATH):
        for file in files:
            Q = deque(maxlen=config.SIZE)
            img = cv2.imread(join(path, file), cv2.IMREAD_UNCHANGED)
            output = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype("float32")

            # Predict
            preds = model.predict(np.expand_dims(img, axis=0))[0]
            Q.append(preds)

            # Getting label
            results = np.array(Q).mean(axis=0)
            i = np.argmax(results)
            label = config.CLASSES[i]

            # Label found
            label_name = basename(basename(path))
            if label_name == label:
                label_found = True
                label_counter += 1

            # draw the activity on the output frame
            text = "activity: {}".format(label)
            cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.25, (0, 255, 0), 5)

            # Make directory
            if not exists(join(config.OUTPUT_PATH, label_name)):
                makedirs(join(config.OUTPUT_PATH, label_name))

            # Save image
            cv2.imwrite(join(config.OUTPUT_PATH, label_name, file), output)

            total_files += 1

        # Average
        if label_found:
            accuracy = (label_counter / total_files) * 100
            accuracy_file.write(basename(basename(path)) + ': ' + str(accuracy) + '%\n\n')
            print(basename(basename(path)) + ': ' + str(accuracy) + '%')

            total_files = 0
            label_counter = 0
            label_found = False


