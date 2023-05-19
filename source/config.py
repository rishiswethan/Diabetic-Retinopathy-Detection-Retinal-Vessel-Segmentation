import os
from torch import cuda
import multiprocessing

MAIN_PATH = str(os.path.dirname(os.path.abspath(__file__)).split('source')[0])  # folder outside source
PROJECT_NAME = str(os.path.dirname(os.path.abspath(__file__)).split('source')[0]).split(os.sep)[-2]  # name of the project, i.e. folder name of the project

windows = (True if (os.name == 'nt') else False)

DEVICE = 'cuda' if cuda.is_available() else 'cpu'

######################################################################################################################################################
# Folders

INPUT_FOLDER = MAIN_PATH + 'input' + os.sep  # inputs parent folder
_DATA_FOLDER = MAIN_PATH + 'data' + os.sep  # data folder
MODELS_FOLDER = MAIN_PATH + 'models' + os.sep  # models folder
DATA_FOLDERS = {
    'org_data': _DATA_FOLDER + 'org_data' + os.sep,  # original dataset folder
    'training_data': _DATA_FOLDER + 'training_data' + os.sep,  # training dataset folder
}

APTOS2019 = "aptos2019-blindness-detection"

DATASETS_IN_USE = [
    APTOS2019
]
TEST_DATASETS = [
    APTOS2019
]

MODEL_SAVE_PATH_BEST_VAL_LOSS = MODELS_FOLDER + 'best_val_loss.pt'
VAL_CALLBACK_OBJ_PATH = MODELS_FOLDER + 'val_callback_obj.pkl'
MODEL_SAVE_PATH_BEST_TRAIN_LOSS = MODELS_FOLDER + 'best_train_loss.pt'
TRAIN_CALLBACK_OBJ_PATH = MODELS_FOLDER + 'train_callback_obj.pkl'

######################################################################################################################################################
# Training parameters

FULL_LABELS = {
    0: 'No_DR',  # 1805 images
    1: 'Mild',  # 370 images
    2: 'Moderate',  # 999 images
    3: 'Severe',  # 193 images
    4: 'Proliferate_DR',  # 295 images
}

# - callback parameters
# below are the parameters for TRAIN lr scheduler
ENABLE_TRAIN_LR_SCHEDULER = True
REDUCE_LR_PATIENCE_TRAIN = 5
REDUCE_LR_FACTOR_TRAIN = 0.5  # setting this to 1.0 will not reduce the learning rate
# below are the parameters for VAL lr scheduler
ENABLE_VAL_LR_SCHEDULER = False
REDUCE_LR_PATIENCE_VAL = 8
REDUCE_LR_FACTOR_VAL = 0.5  # setting this to 1.0 will not reduce the learning rate
# below are the parameters for early stopping
EARLY_STOPPING_MONITOR = 'train_loss'
EARLY_STOPPING_PATIENCE = 20

NUM_CLASSES = len(FULL_LABELS)
TRAIN_TEST_SPLIT = 0.3  # percentage of images to be used for testing

SQUARE_SIZE = 512  # size of the square image

INITIAL_LR = 0.001
INITIAL_EPOCH = 999999

# get unumber of CPUs available
MAX_THREADS = multiprocessing.cpu_count() - 1
