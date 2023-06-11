import os
import torch

######################################################################################################################################################
# data parameters
######################################################################################################################################################
NYU_TEST_SPLIT_PER = 0.2  # test split percentage for nyu dataset
SUN_RGBD_TEST_SPLIT_PER = 0.05  # test split percentage for sun rgbd dataset

######################################################################################################################################################
# training parameters
######################################################################################################################################################
INITIAL_LR = 0.001
BATCH_SIZE = 2
HEIGHT = 1760
WIDTH = 1760
BACKBONE_NAME = "efficientnet-b0"
WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
AUGMENTATION = True
WWO_AUG = False  # train data with and without augmentation
PROB_APPLY_AUGMENTATION = 0.8

# - callback parameters
# below are the parameters for TRAIN lr scheduler
ENABLE_TRAIN_LR_SCHEDULER = True
REDUCE_LR_COOLDOWN_TRAIN = 0
REDUCE_LR_PATIENCE_TRAIN = 4
REDUCE_LR_FACTOR_TRAIN = 0.4  # setting this to 1.0 will not reduce the learning rate
# below are the parameters for VAL lr scheduler
ENABLE_VAL_LR_SCHEDULER = False
REDUCE_LR_COOLDOWN_VAL = 8
REDUCE_LR_PATIENCE_VAL = 8
REDUCE_LR_FACTOR_VAL = 0.5  # setting this to 1.0 will not reduce the learning rate
# below are the parameters for early stopping
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_PATIENCE = 20

######################################################################################################################################################
# classes
######################################################################################################################################################
# Class names
VESSEL = 'vessel'  # Vessel mask
PIXEL_LEVEL = 'pixel_level'  # Pixel level mask when using border masks
BORDER = 'border'  # Border mask when using border masks
UNLABELED = 'unlabeled'

# Mask numbers for each class
UNLABELED_MASK = {"name": UNLABELED, "mask_number": 0}
VESSEL_MASK = {"name": VESSEL, "mask_number": 1}
ALL_ORIGINAL_MASKS = [UNLABELED_MASK, VESSEL_MASK]  # All original masks that are available and in use

MASKS = {
    UNLABELED: 0,
    VESSEL: 1,
}
MASKS_INV = {
    0: UNLABELED,
    1: VESSEL
}
BORDER_THICKNESS = 25  # thickness of the border mask
MIN_PIXELS_MASK_ALL_CLASSES_PER = 0.01  # minimum percentage of pixels that must be present in a mask for it to be sent to train or test data
SPECIAL_DATASET_MIN_PIXELS = 0.05  # same as MIN_PIXELS_MASK_ALL_CLASSES_PER, but for special datasets

# Dataset names
CHASEDB = 'CHASEDB'
DRIVE = 'DRIVE'
STARE = 'STARE'
HRF = 'HRF'
DR_HAGIS = 'DR-HAGIS'
SMDG = 'SMDG'

# Mask types
PIXEL_LEVEL_MASK_TYPE = 'pixel_level_mask_type'  # If CHOSEN_MASK_TYPE is PIXEL_LEVEL, then this is the class name for the pixel level mask
BORDER_MASK_TYPE = 'border_mask_type'

# Split names
TRAIN_FOLDER = 'train' + os.sep
TEST_FOLDER = 'test' + os.sep
TRAIN_TEST_IMAGES_FOLDER = "images" + os.sep
TRAIN_TEST_MASKS_FOLDER = "masks" + os.sep

# Tip: Make sure the mask values are sorted in the dictionary
# Masks to use
CHOSEN_MASKS = [
    UNLABELED,
    VESSEL,
]

# choose which training datasets to use
CHOSEN_TRAINING_DATASETS = [
    CHASEDB,
    DRIVE,
    STARE,
    HRF,
    DR_HAGIS,
    SMDG
]

# chosen mask type
CHOSEN_MASK_TYPE = [PIXEL_LEVEL_MASK_TYPE, BORDER_MASK_TYPE][0]  # choose between PIXEL_LEVEL and BORDER_MASK
TARGET_BORDER_CLASS = VESSEL  # Not used if CHOSEN_MASK_TYPE is PIXEL_LEVEL

# choose special training datasets. These datasets have augmented data, so only one image per folder is used. Values are functions that return the unique folder names
SPECIAL_TRAINING_DATASETS = [
]

TEST_SET_DATASETS = [
    DRIVE
]

USE_STORED_PIXEL_COUNTS = False  # if True, then the pixel counts will be loaded from the file. If False, then the pixel counts will be calculated
SURVEY_PARTIALLY_PIXEL_COUNTS = 1.0

DISALLOW_SPECIAL_DATASETS_IN_TEST = False  # if True, then the special training datasets will not be used in the test data

USE_SPECIAL_DATA_IN_WEIGHTS_CALC = False  # if True, then the special training datasets will be used in the calculation of the weights for each class
DISABLE_BACKGROUND_IN_METRICS = False  # if True, then the background class will be disabled in the metrics

# INFERENCE PARAMETERS
INFERENCE_MODEL = ['BEST_TRAINING_LOSS', 'BEST_VAL_LOSS'][1]  # choose between BEST_TRAINING_LOSS and BEST_VAL_LOSS
PREDICTION_TH = 0.5  # prediction threshold

######################################################################################################################################################
# Do not change below this line. Automatically set parameters according to the above parameters
######################################################################################################################################################
if CHOSEN_MASK_TYPE == BORDER_MASK_TYPE:
    MASKS = {
        UNLABELED: 0,
        BORDER: 1,
        PIXEL_LEVEL: 2,
    }
    MASKS_INV = {
        0: UNLABELED,
        1: BORDER,
        2: PIXEL_LEVEL,
    }
    CHOSEN_MASKS = [
        UNLABELED,
        BORDER,
        PIXEL_LEVEL,
    ]

BINARY_MODE = len(CHOSEN_MASKS) == 2

# chosen masks without unlabeled
CHOSEN_MASKS_NO_UNLABELED = [mask for mask in CHOSEN_MASKS if mask != UNLABELED]
NUM_CLASSES = len(CHOSEN_MASKS_NO_UNLABELED)

# check min pixels classes
if CHOSEN_MASK_TYPE == BORDER_MASK_TYPE:
    CHECK_MIN_PIXELS_CLASSES = [PIXEL_LEVEL]
elif CHOSEN_MASK_TYPE == PIXEL_LEVEL_MASK_TYPE:
    CHECK_MIN_PIXELS_CLASSES = CHOSEN_MASKS_NO_UNLABELED

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
