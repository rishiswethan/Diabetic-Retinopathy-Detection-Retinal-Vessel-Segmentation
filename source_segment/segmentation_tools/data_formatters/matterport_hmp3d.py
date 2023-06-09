import source.segmentation_tools.data_formatters.sun_rgbd as sun_rgbd
import source.segmentation_tools.segmentation_config as seg_cf
import source.config as cf
import source.segmentation_tools.utils as seg_utils
import source.utils as utils

import os
import pickle
import cv2
import shutil


train_image_paths = []
train_mask_paths = []
test_image_paths = []
test_mask_paths = []
num_rejected_masks = 0


def get_all_paths_recursively(folder_name, image_file_extension=".jpg", mask_file_extension=".png", recursively_depth=0, ensure_min_pixels=True, prog_folder_minus=2):
    global train_image_paths, train_mask_paths, test_image_paths, test_mask_paths, num_rejected_masks

    if recursively_depth == 0:
        train_image_paths.clear()
        train_mask_paths.clear()
        test_image_paths.clear()
        test_mask_paths.clear()

    # get all images in folders within images_folder
    for i, file in enumerate(os.listdir(folder_name)):
        if file.endswith(image_file_extension):
            print(file)

            # get paths
            image_path = folder_name + file
            mask_path = folder_name + file.replace(image_file_extension, mask_file_extension)
            mask_path = mask_path.replace(os.sep + "images" + os.sep, os.sep + "masks" + os.sep).replace(image_file_extension, mask_file_extension)

            # make sure image exists
            assert os.path.exists(image_path), "Image does not exist: " + image_path
            assert os.path.exists(mask_path), "Mask does not exist: " + mask_path

            # ensure minimum number of pixels
            if ensure_min_pixels:
                if not seg_utils.check_presence_of_min_pixels_mask(mask_path):
                    num_rejected_masks += 1
                    continue

            # add to train or test
            if (os.sep + "train" + os.sep) in image_path:
                train_image_paths.append(image_path)
                train_mask_paths.append(mask_path)

            elif (os.sep + "test" + os.sep) in image_path:
                test_image_paths.append(image_path)
                test_mask_paths.append(mask_path)

        else:
            # call recursively if folder
            if os.path.isdir(folder_name + file):
                get_all_paths_recursively(folder_name + file + os.sep, image_file_extension, mask_file_extension, recursively_depth + 1, ensure_min_pixels, prog_folder_minus)

    # save paths in pickle format
    if recursively_depth == 0:
        print("Saving paths to pickle file...")
        print("train_image_paths: " + str(len(train_image_paths)))
        print("train_mask_paths: " + str(len(train_mask_paths)))
        print("test_image_paths: " + str(len(test_image_paths)))
        print("test_mask_paths: " + str(len(test_mask_paths)))
        print("num_rejected_masks: " + str(num_rejected_masks))

        for i in range(10):
            print(train_image_paths[i])
            print(train_mask_paths[i])
            print()

        for i in range(10):
            print(test_image_paths[i])
            print(test_mask_paths[i])
            print()

        if not os.path.exists(folder_name + "paths_pkl"):
            os.makedirs(folder_name + "paths_pkl")
        with open(folder_name + "paths_pkl" + os.sep + "paths.pkl", "wb") as f:
            pickle.dump([train_image_paths, train_mask_paths, test_image_paths, test_mask_paths], f)


def get_all_paths_for_pixel(
    target_folder=cf.ORG_DATASET_FOLDERS[seg_cf.HM3D],
    save_folder_name=cf.TRAINING_FOLDER_PIXEL_LEVEL_PATHS[seg_cf.HM3D],
    make_masks_visible=False,
    resize_to=(seg_cf.WIDTH, seg_cf.HEIGHT),
    ensure_min_pixels=False,
):
    # get all mask paths
    with open(target_folder + "paths_pkl" + os.sep + "paths.pkl", "rb") as f:
        train_image_paths, train_mask_paths, test_image_paths, test_mask_paths = pickle.load(f)

    # delete train and test folder if they exist
    if os.path.exists(save_folder_name + "train" + os.sep):
        shutil.rmtree(save_folder_name + "train" + os.sep)
    if os.path.exists(save_folder_name + "test" + os.sep):
        shutil.rmtree(save_folder_name + "test" + os.sep)

    # create train and test folder
    os.makedirs(save_folder_name + "train" + os.sep + "images" + os.sep)
    os.makedirs(save_folder_name + "train" + os.sep + "masks" + os.sep)
    os.makedirs(save_folder_name + "test" + os.sep + "images" + os.sep)
    os.makedirs(save_folder_name + "test" + os.sep + "masks" + os.sep)

    # copy to train and test folder and resize if necessary
    for i, (image_path, mask_path) in enumerate(zip(train_image_paths, train_mask_paths)):
        print("train: " + str(i) + "/" + str(len(train_image_paths)))
        if resize_to:
            image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, resize_to, interpolation=cv2.INTER_NEAREST)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, resize_to, interpolation=cv2.INTER_NEAREST)

            if ensure_min_pixels:
                if not seg_utils.check_presence_of_min_pixels_mask(mask):
                    continue

            cv2.imwrite(save_folder_name + "train" + os.sep + "images" + os.sep + image_path.split(os.sep)[-1], image)
            cv2.imwrite(save_folder_name + "train" + os.sep + "masks" + os.sep + mask_path.split(os.sep)[-1], mask)

        else:
            shutil.copy(image_path, save_folder_name + "train" + os.sep)
            shutil.copy(mask_path, save_folder_name + "train" + os.sep)

    for i, (image_path, mask_path) in enumerate(zip(test_image_paths, test_mask_paths)):
        print("test: " + str(i) + "/" + str(len(test_image_paths)))
        if resize_to:
            image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, resize_to, interpolation=cv2.INTER_NEAREST)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, resize_to, interpolation=cv2.INTER_NEAREST)

            if ensure_min_pixels:
                if not seg_utils.check_presence_of_min_pixels_mask(mask):
                    continue

            cv2.imwrite(save_folder_name + "test" + os.sep + "images" + os.sep + image_path.split(os.sep)[-1], image)
            cv2.imwrite(save_folder_name + "test" + os.sep + "masks" + os.sep + mask_path.split(os.sep)[-1], mask)

        else:
            shutil.copy(image_path, save_folder_name + "test" + os.sep)
            shutil.copy(mask_path, save_folder_name + "test" + os.sep)


if __name__ == '__main__':
    # Run only with CHOSEN_MASK_TYPE = PIXEL_LEVEL_MASK_TYPE. This is to get paths for the original datasets

    # assert seg_cf.CHOSEN_MASK_TYPE == seg_cf.PIXEL_LEVEL_MASK_TYPE, "CHOSEN_MASK_TYPE must be PIXEL_LEVEL_MASK_TYPE"
    # get_all_paths_recursively(folder_name=cf.ORG_DATASET_FOLDERS[seg_cf.HM3D], ensure_min_pixels=True)
    ####################################################################################################################
    # Run with CHOSEN_MASK_TYPE = PIXEL_LEVEL_MASK_TYPE. This is to get paths for the pixel level detection

    # assert seg_cf.CHOSEN_MASK_TYPE == seg_cf.PIXEL_LEVEL_MASK_TYPE, "CHOSEN_MASK_TYPE must be PIXEL_LEVEL_MASK_TYPE"
    # get_all_paths_for_pixel()
    ####################################################################################################################
    # # Run with CHOSEN_MASK_TYPE = BORDER_LEVEL_MASK_TYPE. This is to get paths for training datasets

    # assert seg_cf.CHOSEN_MASK_TYPE == seg_cf.BORDER_MASK_TYPE, "CHOSEN_MASK_TYPE must be BORDER_LEVEL_MASK_TYPE"
    # sun_rgbd.get_borders_for_all_masks(target_folder=cf.ORG_DATASET_FOLDERS[seg_cf.HM3D],
    #                                    save_folder_name=cf.TRAINING_FOLDER_PATHS[seg_cf.HM3D],
    #                                    make_masks_visible=False,
    #                                    resize_to=(seg_cf.WIDTH, seg_cf.HEIGHT))
    ####################################################################################################################
    # # Run with CHOSEN_MASK_TYPE of your choice. This is to get paths for training datasets

    get_all_paths_recursively(folder_name=cf.TRAINING_FOLDER_PATHS[seg_cf.HM3D], ensure_min_pixels=True)
