# import the necessary packages
import os

# initialize the path to the input directory containing our dataset
# of images
DATASET_PATH = "db_images"

# initialize the class labels in the dataset
CLASSES = ["Accidents", "Flood", "Guns", "violence"]

# define the size of the training, validation (which comes from the
# train split), and testing splits, respectively
TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.1
TEST_SPLIT = 0.25

# define the minimum learning rate, maximum learning rate, batch size,
# step size, CLR method, and number of epochs
MIN_LR = 1e-6
MAX_LR = 1e-4
BATCH_SIZE = 32
STEP_SIZE = 8
CLR_METHOD = "triangular"
NUM_EPOCHS = 1

# set the path to the serialized model after training
MODEL_PATH = os.path.sep.join(["/output", "SGD_VGG16", "camera.model"])

# define the path to the output learning rate finder plot, training
# history plot and cyclical learning rate plot
LRFIND_PLOT_PATH = os.path.sep.join(["/output", "SGD_VGG16", "lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["/output", "SGD_VGG16", "training_plot.png"])
CLR_PLOT_PATH = os.path.sep.join(["/output", "SGD_VGG16", "clr_plot.png"])

#INPUT_PATH = os.path.sep.join(["videos", "video_accidents.mp4"])
#OUTPUT_PATH = os.path.sep.join(["output4", "video_accidents.mp4"])
INPUT_PATH = "test_images"
OUTPUT_PATH = "test_images_out"
LOG_ACCURACY = os.path.sep.join(["/output", "SGD_VGG16", "accuracy.txt"])
SIZE = 128