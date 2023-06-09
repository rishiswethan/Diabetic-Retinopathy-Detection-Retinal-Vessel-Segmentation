import os
import source.segmentation_tools.segmentation_config as seg_cf

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

P2E_INPUT_FOLDER = INPUT_FOLDER + 'stitch_images' + ls  # inputs for multi perspective stitching
FLOOR_DETECT_INPUT_FOLDER = INPUT_FOLDER + 'floor_detect_images' + ls  # inputs for floor detection
POINT_CLOUD_INPUT_FOLDER = INPUT_FOLDER + 'point_cloud' + ls  # inputs for point cloud
FLOOR_DETECT_360_INPUT_FOLDER = INPUT_FOLDER + 'floor_detect_360_input' + ls  # inputs for floor detection in 360
TILE_COLOUR_CHANGE_BASE_INPUT_FOLDER = os.path.join(INPUT_FOLDER, 'tile_colour_change', 'base_tile')  # inputs for tile colour change
TILE_COLOUR_CHANGE_TARGET_INPUT_FOLDER = os.path.join(INPUT_FOLDER, 'tile_colour_change', 'target_colour_tile')  # inputs for tile colour change

######################################################################################################################################################
# 360 and other related paths
######################################################################################################################################################
C2E_OUTPUT_FOLDER = OUTPUT_FOLDER + 'c2e' + ls  # outputs folder for c2e
E2C_OUTPUT_FOLDER = OUTPUT_FOLDER + 'e2c' + ls  # output folder of e2c
E2P_OUTPUT_FOLDER = OUTPUT_FOLDER + 'e2p' + ls  # output folder of e2p
P2E_OUTPUT_FOLDER = OUTPUT_FOLDER + 'p2e' + ls  # output folder of p2e
BORDER_DETECT_OUTPUT_FOLDER = OUTPUT_FOLDER + 'border_detect' + ls  # output folder of border detect
FLOOR_DETECT_OUTPUT_FOLDER = OUTPUT_FOLDER + 'floor_detect' + ls  # output folder of floor detect
POINT_CLOUD_OUTPUT_FOLDER = OUTPUT_FOLDER + 'point_cloud' + ls  # output folder of point cloud
TILE_COLOUR_CHANGE_OUTPUT_FOLDER = OUTPUT_FOLDER + 'tile_colour_change' + ls  # output folder of tile colour change
BUFFER_FOLDER = OUTPUT_FOLDER + 'buffer' + ls  # buffer folder

C2E_OUTPUT_IMAGE_PATH = C2E_OUTPUT_FOLDER + 'c2e_output.png'  # output image of c2e
E2C_OUTPUT_IMAGE_PREFIX = 'e2c_output'  # Add file extension or <image_identifier>.<extension> to this path
E2C_OUTPUT_IMAGE_PATH = E2C_OUTPUT_FOLDER + E2C_OUTPUT_IMAGE_PREFIX  # output image of e2c
E2P_OUTPUT_IMAGE_PATH = E2P_OUTPUT_FOLDER + 'e2p_output.jpg'  # output image of e2p
BORDER_DETECT_OUTPUT_PATH = BORDER_DETECT_OUTPUT_FOLDER + 'border_detect_output.jpg'  # output image of border detect
BORDER_PIXELS_CSV_OUTPUT_PATH = BORDER_DETECT_OUTPUT_FOLDER + 'border_pixels.csv'  # output csv of border detect

PERSPECTIVE_CONFIG_JSON_PATH = INPUT_FOLDER + 'perspective_config.json'  # perspective config json

######################################################################################################################################################
# Segmentation
######################################################################################################################################################
# Primary paths
_DATA_FOLDER = MAIN_PATH + 'data' + ls  # data folder
_ORIGINAL_DATASET_PRIMARY_FOLDER = "/mnt/Extra/Programming/AI/Datasets/segmentation/floor_detect/Data used for training/original_data/"  # original dataset folder
_TRAINING_DATASET_PRIMARY_FOLDER = _DATA_FOLDER + 'training_data' + ls  # training dataset folder
TRAINED_MODEL_FOLDER = MAIN_PATH + 'trained_models' + ls  # trained model folder

MODEL_SAVE_PATH_BEST_VAL_LOSS = TRAINED_MODEL_FOLDER + 'model_best_val_loss.pth'  # model save path
MODEL_SAVE_PATH_BEST_TRAIN_LOSS = TRAINED_MODEL_FOLDER + 'model_best_train_loss.pth'  # model save path
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

