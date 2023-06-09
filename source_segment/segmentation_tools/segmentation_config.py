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
BATCH_SIZE = 32
HEIGHT = 320
WIDTH = 320
BACKBONE_NAME = "resnet152"
WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
AUGMENTATION = True
WWO_AUG = False  # train data with and without augmentation
PROB_APPLY_AUGMENTATION = 0.8

# - callback parameters
# below are the parameters for TRAIN lr scheduler
ENABLE_TRAIN_LR_SCHEDULER = True
REDUCE_LR_COOLDOWN_TRAIN = 0
REDUCE_LR_PATIENCE_TRAIN = 2
REDUCE_LR_FACTOR_TRAIN = 0.4  # setting this to 1.0 will not reduce the learning rate
# below are the parameters for VAL lr scheduler
ENABLE_VAL_LR_SCHEDULER = False
REDUCE_LR_COOLDOWN_VAL = 4
REDUCE_LR_PATIENCE_VAL = 4
REDUCE_LR_FACTOR_VAL = 0.5  # setting this to 1.0 will not reduce the learning rate
# below are the parameters for early stopping
EARLY_STOPPING_MONITOR = 'train_loss'
EARLY_STOPPING_PATIENCE = 20

######################################################################################################################################################
# classes
######################################################################################################################################################
# Class names
FLOOR = 'floor'  # Floor mask when using pixel level masks
WALL = 'wall'  # Wall mask when using pixel level masks
CEILING = 'ceiling'  # Ceiling mask when using pixel level masks
PIXEL_LEVEL = 'pixel_level'  # Pixel level mask when using border masks
BORDER = 'border'  # Border mask when using border masks
UNLABELED = 'unlabeled'

# Mask numbers for each class
UNLABELED_MASK = {"name": UNLABELED, "mask_number": 0}
FLOOR_MASK = {"name": FLOOR, "mask_number": 1}
WALL_MASK = {"name": WALL, "mask_number": 2}
# CEILING_MASK = {"name": CEILING, "mask_number": 3}

ALL_ORIGINAL_MASKS = [UNLABELED_MASK, FLOOR_MASK, WALL_MASK]  # All original masks that are available and in use

MASKS = {
    UNLABELED: 0,
    FLOOR: 1,
    WALL: 2,
}
MASKS_INV = {
    0: UNLABELED,
    1: FLOOR,
    2: WALL,
}
BORDER_THICKNESS = 25  # thickness of the border mask
MIN_PIXELS_MASK_ALL_CLASSES_PER = 0.01  # minimum percentage of pixels that must be present in a mask for it to be sent to train or test data
SPECIAL_DATASET_MIN_PIXELS = 0.05  # same as MIN_PIXELS_MASK_ALL_CLASSES_PER, but for special datasets

# Dataset names
NYU = 'nyu'
SCENE_NET = 'scene_net'
SYNTHETIC_DATA = 'synthetic_data'
ADE20K = 'aed20k'
SUN_RGBD = 'sun_rgbd'
HM3D = 'hm3d'
REPLICA = 'replica'
HM3D_CUST_V = 'hm3d_cust_v'

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
    FLOOR,
    # WALL,
]

# choose which training datasets to use
CHOSEN_TRAINING_DATASETS = [
    # NYU,
    # SCENE_NET,
    ADE20K,  # This is a benchmark dataset
    SUN_RGBD,  # This one has NYU data and some other datasets as well
    HM3D_CUST_V,
    REPLICA
]

# chosen mask type
CHOSEN_MASK_TYPE = [PIXEL_LEVEL_MASK_TYPE, BORDER_MASK_TYPE][0]  # choose between PIXEL_LEVEL and BORDER_MASK
TARGET_BORDER_CLASS = FLOOR_MASK  # Not used if CHOSEN_MASK_TYPE is PIXEL_LEVEL

# choose special training datasets. These datasets have augmented data, so only one image per folder is used. Values are functions that return the unique folder names
SPECIAL_TRAINING_DATASETS = [
    SCENE_NET,
    HM3D_CUST_V
]

TEST_SET_DATASETS = [
    ADE20K
]

USE_STORED_PIXEL_COUNTS = False  # if True, then the pixel counts will be loaded from the file. If False, then the pixel counts will be calculated
SURVEY_PARTIALLY_PIXEL_COUNTS = 1.0

DISALLOW_SPECIAL_DATASETS_IN_TEST = False  # if True, then the special training datasets will not be used in the test data

USE_SPECIAL_DATA_IN_WEIGHTS_CALC = False  # if True, then the special training datasets will be used in the calculation of the weights for each class
DISABLE_BACKGROUND_IN_METRICS = False  # if True, then the background class will be disabled in the metrics

######################################################################################################################################################
# Inference parameters

INFERENCE_MODEL = ['BEST_TRAINING_LOSS', 'BEST_VAL_LOSS'][1]  # choose between BEST_TRAINING_LOSS and BEST_VAL_LOSS
EXTRACTION_MODE_360 = ["cubemap", "perspective", "simple_crop"][1]  # choose between cubemap and perspective for 360 images extraction

# 360 prediction parameters
PERSPECTIVE_SQUARE_SIZE = 512  # size of the square image that will be extracted from the perspective extraction
INFERENCE_360_SKIP_ROOF = True  # if True, then the roof will be skipped in the inference
BLUR_FACTOR = 10  # Blur will range from 0 to BLUR_FACTOR

# First main fov parameters
PREDICTION_TH = 0.8  # threshold for the prediction mask and primary fov
INFERENCE_360_PERSPECTIVE_FOV_MAIN = 75  # main fov for the perspective extraction. Default viewing angle is always 0 or 360
INFERENCE_360_PERSPECTIVE_MAIN_H_INC = 10  # horizontal increment by this angle for the main fov
INFERENCE_MAIN_OVERLAP_TH = 128  # overlap threshold for the main fov. Setting it higher will mean more predictions
INFERENCE_360_V_VIEWING_ANGLE_MAIN = 360  # vertical viewing angle for the perspective extraction during main fov

# Max fov parameters (used to extract content missed by the main fov)
PREDICTION_TH_MAX_FOV = 0.8
INFERENCE_360_PERSPECTIVE_FOV_MAX = 140
INFERENCE_360_PERSPECTIVE_MAX_H_INC = 10
INFERENCE_MAX_OVERLAP_TH = 128
INFERENCE_360_V_VIEWING_ANGLE_MAX = 360

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

# check min pixels classes
if CHOSEN_MASK_TYPE == BORDER_MASK_TYPE:
    CHECK_MIN_PIXELS_CLASSES = [PIXEL_LEVEL]
elif CHOSEN_MASK_TYPE == PIXEL_LEVEL_MASK_TYPE:
    CHECK_MIN_PIXELS_CLASSES = CHOSEN_MASKS_NO_UNLABELED

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
