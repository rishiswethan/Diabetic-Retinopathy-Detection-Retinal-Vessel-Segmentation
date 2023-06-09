import os
import source_segment.segmentation_tools.segmentation_config as seg_cf

windows = (True if (os.name == 'nt') else False)
if windows:
    OS = 'windows'
    ls = '\\'  # Windows uses backslash
else:
    OS = 'linux'
    ls = '/'  # Linux and mac uses forward slash

######################################################################################################################################################
MAIN_PATH = str(os.path.dirname(os.path.abspath(__file__)).split('source')[0])  # folder outside source
PROJECT_NAME = 'FloorReplace'

INPUT_FOLDER = MAIN_PATH + 'input' + ls  # inputs parent folder
OUTPUT_FOLDER = MAIN_PATH + 'detailed_output' + ls  # outputs parent folder
SIMPLE_OUTPUT_FOLDER = MAIN_PATH + 'output' + ls  # simple outputs parent folder

######################################################################################################################################################
# Segmentation
######################################################################################################################################################
# Primary paths
_DATA_FOLDER = MAIN_PATH + 'data' + ls  # data folder
_ORIGINAL_DATASET_PRIMARY_FOLDER = MAIN_PATH + ''  # original dataset folder
_TRAINING_DATASET_PRIMARY_FOLDER = _DATA_FOLDER + 'training_data' + ls  # training dataset folder
TRAINED_MODEL_FOLDER = MAIN_PATH + 'trained_models' + ls  # trained model folder

MODEL_SAVE_PATH_BEST_VAL_LOSS = TRAINED_MODEL_FOLDER + 'segment_best_val_loss.pth'  # model save path
MODEL_SAVE_PATH_BEST_TRAIN_LOSS = TRAINED_MODEL_FOLDER + 'segment_best_train_loss.pth'  # model save path
PIXEL_COUNT_PATH = TRAINED_MODEL_FOLDER + 'pixel_count.pkl'  # pixel count path
TRAINING_LOG_PATH = TRAINED_MODEL_FOLDER + 'training_log.csv'  # training log path
TRAIN_CALLBACK_OBJ_PATH = TRAINED_MODEL_FOLDER + 'train_callback_obj.pkl'  # training callback object path
VAL_CALLBACK_OBJ_PATH = TRAINED_MODEL_FOLDER + 'val_callback_obj.pkl'  # validation callback object path

# Original dataset paths
ORG_DATASET_FOLDERS = {
    seg_cf.NYU: _ORIGINAL_DATASET_PRIMARY_FOLDER + 'nyu' + ls,  # nyu data folder
    seg_cf.SCENE_NET: _ORIGINAL_DATASET_PRIMARY_FOLDER + 'SceneNet' + ls + 'scenenetrgbd' + ls,  # segmentation network folder
    seg_cf.SYNTHETIC_DATA: _ORIGINAL_DATASET_PRIMARY_FOLDER + 'SyntheticData' + ls,  # synthetic data folder
    seg_cf.ADE20K: _ORIGINAL_DATASET_PRIMARY_FOLDER + 'ADE20K' + ls,  # ADE20K data folder
    seg_cf.SUN_RGBD: _ORIGINAL_DATASET_PRIMARY_FOLDER + 'SUNRGBD' + ls + 'SUNRGBD' + ls,  # SUNRGBD data folder
    seg_cf.HM3D: _ORIGINAL_DATASET_PRIMARY_FOLDER + 'hm3d' + ls,  # HM3D data folder
    seg_cf.REPLICA: _ORIGINAL_DATASET_PRIMARY_FOLDER + 'replica' + ls,  # replica data folder
    seg_cf.HM3D_CUST_V: _ORIGINAL_DATASET_PRIMARY_FOLDER + 'hm3d_pano' + ls + "custom_view_angle",  # hm3d custom v data folder
}

# Training dataset paths for pixel level segmentation
TRAINING_FOLDER_PIXEL_LEVEL_PATHS = {
    seg_cf.NYU: _TRAINING_DATASET_PRIMARY_FOLDER + seg_cf.NYU + ls,  # nyu training data folder
    seg_cf.SCENE_NET: _TRAINING_DATASET_PRIMARY_FOLDER + seg_cf.SCENE_NET + ls,  # segmentation network training data folder
    seg_cf.SYNTHETIC_DATA: _TRAINING_DATASET_PRIMARY_FOLDER + seg_cf.SYNTHETIC_DATA + ls,  # synthetic data training data folder
    seg_cf.ADE20K: _TRAINING_DATASET_PRIMARY_FOLDER + seg_cf.ADE20K + ls,  # ADE20K training data folder
    seg_cf.SUN_RGBD: _TRAINING_DATASET_PRIMARY_FOLDER + seg_cf.SUN_RGBD + ls,  # SUNRGBD training data folder
    seg_cf.HM3D: _TRAINING_DATASET_PRIMARY_FOLDER + seg_cf.HM3D + ls,  # HM3D training data folder.
    seg_cf.REPLICA: _TRAINING_DATASET_PRIMARY_FOLDER + seg_cf.REPLICA + ls,  # replica training data folder.
    seg_cf.HM3D_CUST_V: _TRAINING_DATASET_PRIMARY_FOLDER + seg_cf.HM3D_CUST_V + ls,  # hm3d custom v training data folder.
}

# Training dataset floor border paths
TRAINING_FOLDER_BORDER_LEVEL_PATHS = {
    seg_cf.NYU: _TRAINING_DATASET_PRIMARY_FOLDER + seg_cf.NYU + "_border_floor" + ls,  # nyu training data floor border folder
    seg_cf.SCENE_NET: _TRAINING_DATASET_PRIMARY_FOLDER + seg_cf.SCENE_NET + ls,  # segmentation network training data floor border folder. We use the same folder as pixel level segmentation. Border is rendered on the fly
    seg_cf.SYNTHETIC_DATA: _TRAINING_DATASET_PRIMARY_FOLDER + seg_cf.SYNTHETIC_DATA + "_border_floor" + ls,  # synthetic data training data floor border folder
    seg_cf.ADE20K: _TRAINING_DATASET_PRIMARY_FOLDER + seg_cf.ADE20K + "_border_floor" + ls,  # ADE20K training data floor border folder
    seg_cf.SUN_RGBD: _TRAINING_DATASET_PRIMARY_FOLDER + seg_cf.SUN_RGBD + "_border_floor" + ls,  # SUNRGBD training data floor border folder
    seg_cf.HM3D: _TRAINING_DATASET_PRIMARY_FOLDER + seg_cf.HM3D + "_border_floor" + ls,  # HM3D training data floor border folder
    seg_cf.REPLICA: _TRAINING_DATASET_PRIMARY_FOLDER + seg_cf.REPLICA + "_border_floor" + ls,  # replica training data floor border folder
}

# path lists names
PIXEL_LEVEL_MASKS_PATHS = 'paths.pkl'  # pixel level masks

######################################################################################################################################################
# Automatic changes, do not change below this line
######################################################################################################################################################
# change training dataset paths if MASK_TYPE is not pixel
if seg_cf.CHOSEN_MASK_TYPE == seg_cf.BORDER_MASK_TYPE:
    TRAINING_FOLDER_PATHS = TRAINING_FOLDER_BORDER_LEVEL_PATHS
elif seg_cf.CHOSEN_MASK_TYPE == seg_cf.PIXEL_LEVEL_MASK_TYPE:
    TRAINING_FOLDER_PATHS = TRAINING_FOLDER_PIXEL_LEVEL_PATHS


# change model save path if inference model is not best val loss, which is the default
if seg_cf.INFERENCE_MODEL == 'BEST_TRAINING_LOSS':
    MODEL_SAVE_PATH = MODEL_SAVE_PATH_BEST_TRAIN_LOSS
else:
    MODEL_SAVE_PATH = MODEL_SAVE_PATH_BEST_VAL_LOSS

assert os.path.exists(_ORIGINAL_DATASET_PRIMARY_FOLDER), "Original dataset folder does not exist, please check the path"
# # class and original image h5 names
# train_images = 'images.h5'  # train images h5 name

