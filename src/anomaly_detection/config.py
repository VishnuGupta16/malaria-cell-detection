from pathlib import Path

# Absolute path to the project's datasets/ folder, built from this file's
# location so it works no matter where you launch the program from.
# config.py is at src/anomaly_detection/config.py, so the project root is 3 levels up.
DATA_DIR = str(Path(__file__).resolve().parents[2] / "datasets")

# To enable visual of small or 1 dataset
VISUALIZE = False

# How many images to work with. 0 = use the full dataset.
# Set a positive number (e.g. 2000) to work on a smaller subset and speed things up.
BATCH_SIZE = 0

TRAIN_SET_SIZE = 0.7 # Relative size of the training set
VAL_SET_SIZE = 0.15  # Relative size of the validation set
TEST_SIZE = 0.15     # Relative size of the test set

IMAZE_SIZE = [64, 64] # 64*64 pixel 
MODEL_INPUT_SHAPE = (64 , 64, 3) # 64*64 pixel and 3 colour RBG
CLASSES = 2