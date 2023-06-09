import source_segment.config as cf
import h5py
import os
import numpy as np
import cv2
# import cv2.cv2 as cv2
from PIL import Image
import pickle

import source_segment.segmentation_tools.segmentation_config as seg_cf
import source_segment.utils as utils
import source_segment.segmentation_tools.utils as seg_utils


def get_all_images(images_folder, mask_number=None, file_extension=".ppm"):
    images_as_arrays = {}
    cnt = 0
    for folder in os.listdir(images_folder):
        # only folders
        if not os.path.isdir(images_folder + folder):
            continue

        # get all images in folders within images_folder
        for file in os.listdir(images_folder + folder):
            if file.endswith(file_extension):
                # print(folder, file)
                img = cv2.imread(images_folder + folder + os.sep + file, cv2.IMREAD_GRAYSCALE)
                # img = make_img_binary(img, binary_one_replacement=mask_number, binary_th=230)
                img[img < 230] = 0
                img[img != 0] = mask_number

                images_as_arrays[file] = img
                cnt += 1

    assert len(images_as_arrays) == cnt

    print("Total images:", len(images_as_arrays))

    return images_as_arrays


def combine_masks(masks_lists):
    combined_masks = {}

    mask_list_1 = masks_lists[0]
    mask_list_2 = masks_lists[1]
    for file_name in mask_list_1:
        combined_masks[file_name] = np.add(mask_list_1[file_name], mask_list_2[file_name])

    return combined_masks


def recurrently_reformat_images(images_folder, target_ext="ppm", new_ext="jpg"):
    for file in os.listdir(images_folder):
        if os.path.isdir(images_folder + file):
            recurrently_reformat_images(images_folder + file + os.sep, target_ext, new_ext)
        else:
            if file.endswith(target_ext):
                Image.open(images_folder + file).save(images_folder + file.replace(target_ext, new_ext))


def format_nyu_data():
    # get floor masks
    nyu_floor_masks = get_all_images(cf.ORG_DATASET_FOLDERS[seg_cf.NYU] + "MaskedImagesFilteredFloor" + os.sep, mask_number=seg_cf.MASKS[seg_cf.FLOOR])

    # get wall masks
    nyu_wall_masks = get_all_images(cf.ORG_DATASET_FOLDERS[seg_cf.NYU] + "MaskedImagesFilteredWall" + os.sep, mask_number=seg_cf.MASKS[seg_cf.WALL])

    assert len(nyu_floor_masks) == len(nyu_wall_masks)

    # combine masks and save
    combined_masks = combine_masks([nyu_floor_masks, nyu_wall_masks])

    # reformat and save images
    recurrently_reformat_images(cf.ORG_DATASET_FOLDERS[seg_cf.NYU] + "originalImages" + os.sep, target_ext="ppm", new_ext="jpg")

    # split into train and test
    combined_masks_list = list(combined_masks.keys())
    combined_masks_list = seg_utils.shuffle_train_data(combined_masks_list)
    test_masks_file_names = combined_masks_list[:int(len(combined_masks_list) * seg_cf.NYU_TEST_SPLIT_PER)]
    train_masks_file_names = combined_masks_list[int(len(combined_masks_list) * seg_cf.NYU_TEST_SPLIT_PER):]

    # save masks
    for file_names in [train_masks_file_names, test_masks_file_names]:
        for file in file_names:
            if file_names == train_masks_file_names:
                cv2.imwrite(cf.TRAINING_FOLDER_PATHS[seg_cf.NYU] + os.sep + "train" + os.sep + "masks" + os.sep + file.replace("ppm", "png"), combined_masks[file])
            else:
                cv2.imwrite(cf.TRAINING_FOLDER_PATHS[seg_cf.NYU] + os.sep + "test" + os.sep + "masks" + os.sep + file.replace("ppm", "png"), combined_masks[file])


# this function also filters images with bad masks
def get_all_image_mask_paths(folder_path=cf.TRAINING_FOLDER_PATHS[seg_cf.NYU]):
    train_image_paths = []
    train_mask_paths = []
    test_image_paths = []
    test_mask_paths = []
    for mode in ["train", "test"]:
        for type in ["images", "masks"]:
            folder = os.path.join(folder_path, mode, type)
            for file in os.listdir(folder):
                if type == "images":
                    mask_path = os.path.join(folder_path, mode, "masks", file.replace(file.split(".")[-1], "png"))
                    if seg_utils.check_presence_of_min_pixels_mask(mask_path_or_arr=mask_path):
                        if mode == "train":
                            train_image_paths.append(os.path.join(folder, file))
                            train_mask_paths.append(mask_path)
                        else:
                            test_image_paths.append(os.path.join(folder, file))
                            test_mask_paths.append(mask_path)

    print("train_image_paths:", len(train_image_paths))
    print("train_mask_paths:", len(train_mask_paths))
    print("test_image_paths:", len(test_image_paths))
    print("test_mask_paths:", len(test_mask_paths))

    # save paths in a pickle file
    with open(cf.TRAINING_FOLDER_PATHS[seg_cf.NYU] + os.sep + "paths_pkl" + os.sep + "paths.pkl", "wb") as f:
        pickle.dump([train_image_paths, train_mask_paths, test_image_paths, test_mask_paths], f)


def get_borders_for_all_masks(folder_path=cf.TRAINING_FOLDER_PATHS[seg_cf.NYU], save_folder_name="borders"):
    def get_border_and_combine(mask_path):
        mask_arr = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_numbers = list(seg_cf.MASKS.values())
        mask_numbers.remove(0)

        border_masks = seg_utils.make_border_of_masks(
                            mask_arr=mask_arr,
                            mask_numbers=mask_numbers
                        )
        border_masks = np.array(border_masks)
        return border_masks

    # get all mask paths
    with open(cf.TRAINING_FOLDER_PATHS[seg_cf.NYU] + os.sep + "paths_pkl" + os.sep + "paths.pkl", "rb") as f:
        train_image_paths, train_mask_paths, test_image_paths, test_mask_paths = pickle.load(f)

    # get all masks
    train_border_mask_paths = []
    for mask_path in train_mask_paths:
        save_path = mask_path.replace(os.sep + "masks" + os.sep, os.sep + save_folder_name + os.sep)
        save_folder_path = os.path.dirname(save_path)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        borders_mask = get_border_and_combine(mask_path)
        train_border_mask_paths.append(save_path)
        cv2.imwrite(save_path, borders_mask)

    test_border_mask_paths = []
    for mask_path in test_mask_paths:
        save_path = mask_path.replace(os.sep + "masks" + os.sep, os.sep + save_folder_name + os.sep)
        save_folder_path = os.path.dirname(save_path)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        borders_mask = get_border_and_combine(mask_path)
        test_border_mask_paths.append(save_path)
        cv2.imwrite(save_path, borders_mask)


    # save borders in a pickle file
    with open(cf.TRAINING_FOLDER_PATHS[seg_cf.NYU] + os.sep + "paths_pkl" + os.sep + "paths_borders.pkl", "wb") as f:
        pickle


if __name__ == "__main__":
    # format_nyu_data()
    get_all_image_mask_paths()
