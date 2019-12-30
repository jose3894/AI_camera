# import the necessary packages
import os

# initialize the path to the input directory containing our dataset
# of images
DATASET_PATH = "db_images2"

# initialize the class labels in the dataset
CLASSES = ["Flood", "Robbery", "Violence"]

# define the size of the training, validation (which comes from the
# train split), and testing splits, respectively
TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.1
TEST_SPLIT = 0.25

# define the minimum learning rate, maximum learning rate, batch size,
# step size, CLR method, and number of epochs
MIN_LR = 10**(-5.5)
MAX_LR = 10**(-3.5)
BATCH_SIZE = 32
STEP_SIZE = 8
CLR_METHOD = "triangular"
NUM_EPOCHS = 48

# set the path to the serialized model after training
MODEL_PATH = os.path.sep.join(["output3", "camera.model"])

# define the path to the output learning rate finder plot, training
# history plot and cyclical learning rate plot
LRFIND_PLOT_PATH = os.path.sep.join(["output3", "lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["output3", "training_plot.png"])
CLR_PLOT_PATH = os.path.sep.join(["output3", "clr_plot.png"])

#INPUT_PATH = os.path.sep.join(["videos", "fort_mcmurray_wildfire.mp4"])
#OUTPUT_PATH = os.path.sep.join(["output3", "output3.mp4"])
INPUT_PATH = "images"
OUTPUT_PATH = "images_out"