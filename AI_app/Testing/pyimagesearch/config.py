# import the necessary packages
import os

CLASSES = ["Accidents", "Flood", "Guns", "violence"]
INPUT_PATH = os.path.sep.join(["Testing", "test_images"])
OUTPUT_PATH = os.path.sep.join(["Testing", "test_images_out"])
LOG_ACCURACY = os.path.sep.join(["Testing", "model_accuracy", "Adam_VGG16", "accuracy.txt"])
MODEL_PATH_TEST = os.path.sep.join(["/app", "Model", "AI_camera_model", "Adam_VGG16", "camera.model"])
SIZE = 128
