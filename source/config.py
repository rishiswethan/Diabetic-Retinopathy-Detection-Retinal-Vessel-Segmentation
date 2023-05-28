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
IDRID = "IDRiD"
EyePACS = "EyePACS"
EyePACS_test = "EyePACS_test"  # there are more test images than train images in EyePACS
SUSTech = "SUSTech"

DATASETS_IN_USE = [
    APTOS2019,
    IDRID,
    SUSTech,
    # EyePACS,
    # EyePACS_test
]
TEST_DATASETS = [
    APTOS2019,
    IDRID,
    SUSTech,
    # EyePACS,
    # EyePACS_test
]

MODEL_SAVE_PATH_BEST_VAL_LOSS = MODELS_FOLDER + 'best_val_loss.pt'
VAL_CALLBACK_OBJ_PATH = MODELS_FOLDER + 'val_callback_obj.pkl'
MODEL_SAVE_PATH_BEST_TRAIN_LOSS = MODELS_FOLDER + 'best_train_loss.pt'
TRAIN_CALLBACK_OBJ_PATH = MODELS_FOLDER + 'train_callback_obj.pkl'

BEST_HP_JSON_SAVE_PATH = MODELS_FOLDER + 'best_hp.json'
TUNER_CSV_SAVE_PATH = MODELS_FOLDER + 'tuner.csv'
TUNER_SAVE_PATH = MODELS_FOLDER + 'tuner.pkl'

TRAIN_TUNE_TARGET = 'val_acc'  # metric to tune for
TRAIN_TUNE_MODE = ['max', 'min'][0]  # acc is to be maximized, loss minimized, etc

######################################################################################################################################################
# Training parameters

_TUNING_MODELS_LIST = [
    # 'resnet18',
    # 'resnet34',
    # 'resnet50',
    # 'resnet101',
    # 'resnet152',
    # 'inception',
    # 'eff_b0',
    # 'eff_b1',
    # 'eff_b2',
    'eff_b3',
    # 'eff_b4',
    # 'eff_b5',
    # 'eff_v2_s',
    # 'convnext_t',
    # 'mobilenet_v3_small',
    # 'mobilenet_v3_large',
    # 'vit_b_16'
]
TUNE_HP_RANGES = {
    'batch_size': (
        [8],
        'choice'),

    'prob_apply_augmentation': (
        [0.9],
        'choice'),

    'reduce_lr_factor_val': (
        [0.5],
        'choice'),

    'reduce_lr_patience_val': (
        [10],
        'choice'),

    'reduce_lr_factor_train': (
        [0.2],
        'choice'),

    'reduce_lr_patience_train': (
        [5],
        'choice'),

    'use_geometric_augmentation': (
        [True, False],
        'choice'),

    'use_colour_augmentation': (
        [True, False],
        'choice'),

    'conv_model': (_TUNING_MODELS_LIST, 'choice'),
}
TUNING_EARLY_STOPPING_PATIENCE = 10

# must be in ascending order
FULL_LABELS = {
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferate_DR',
}

# - callback parameters
# below are the parameters for TRAIN lr scheduler
ENABLE_TRAIN_LR_SCHEDULER = True
REDUCE_LR_PATIENCE_TRAIN = 5
REDUCE_LR_FACTOR_TRAIN = 0.5  # setting this to 1.0 will not reduce the learning rate
# below are the parameters for VAL lr scheduler
ENABLE_VAL_LR_SCHEDULER = True
REDUCE_LR_PATIENCE_VAL = 2
REDUCE_LR_FACTOR_VAL = 0.2  # setting this to 1.0 will not reduce the learning rate
# below are the parameters for early stopping
EARLY_STOPPING_MONITOR = 'val_acc'  # 'val_acc', 'val_loss', 'train_acc', 'train_loss'
EARLY_STOPPING_MONITOR_MODE = ['max', 'min'][0]
EARLY_STOPPING_PATIENCE = 20

NUM_CLASSES = len(FULL_LABELS)
TRAIN_TEST_SPLIT = 0.3  # percentage of images to be used for testing

SQUARE_SIZE = 512  # size of the square image

INITIAL_LR = 0.001
INITIAL_EPOCH = 999999

# get number of CPUs available
MAX_THREADS = 7

######################################################################################################################################################

# max_trails for tuning is currently set to the number of combinations of the above hyperparameters
MAX_TRIALS = 1
for key in TUNE_HP_RANGES.keys():
    if TUNE_HP_RANGES[key][1] == 'range':
        MAX_TRIALS *= TUNE_HP_RANGES[key][0][2]
    else:
        MAX_TRIALS *= len(TUNE_HP_RANGES[key][0])
