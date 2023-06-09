import json
import os
import numpy as np
import shutil
import cv2
import pickle
import random
from PIL import Image

import source_segment.config as cf
import source_segment.segmentation_tools.segmentation_config as seg_cf
import source_segment.segmentation_tools.utils as seg_utils
import source_segment.utils as utils

CHOSEN_MASKS = seg_cf.CHOSEN_MASKS
CHOSEN_MASKS.remove(seg_cf.UNLABELED)

relevant_masks = {
    seg_cf.FLOOR: [
        "floor",
        "flooring",
        "carpet",
        "rug",
    ],
    seg_cf.WALL: [
        "wall",
    ]
}
relevant_mask_list = []
for rel_mask in relevant_masks:
    if rel_mask in CHOSEN_MASKS:
        print("rel_mask:", rel_mask, "relevant_masks[rel_mask]:", relevant_masks[rel_mask])
        for mask in relevant_masks[rel_mask]:
            relevant_mask_list.append(mask)
print("relevant_mask_list:", relevant_mask_list)


def load_json(file_name):
    with open(file_name) as f:
        data = dict(json.load(f))
    return data


def _get_relevant_json_data(data, relevant_masks=relevant_mask_list):
    mask_ids_pairs = {}
    for mask in relevant_masks:
        mask_ids_pairs[mask] = []

    for elm in dict(data)['annotation']['object']:
        for mask in relevant_masks:
            if mask in elm['name']:
                mask_ids_pairs[mask].append(elm['id'])

    return mask_ids_pairs


def _get_mask_file_name(folder_name, mask_id):
    # ADE_train_00001884
    # instance_000_ADE_train_00001884.png
    return "instance_" + str(mask_id).zfill(3) + "_" + folder_name.split(os.sep)[-2] + ".png"


def get_all_images_recurrently(folder_name, recurrent_depth=0, relevant_masks=CHOSEN_MASKS, save_folder=cf.TRAINING_FOLDER_PIXEL_LEVEL_PATHS[seg_cf.ADE20K]):
    # print("folder_name:", folder_name.split(cf.MAIN_PATH)[1])

    # clear or create save folders
    if recurrent_depth == 0:
        if os.path.exists(save_folder + "images" + os.sep):
            utils.delete_folder_contents(save_folder + "images" + os.sep)
        else:
            os.makedirs(save_folder + "images" + os.sep)
        if os.path.exists(save_folder + "masks" + os.sep):
            utils.delete_folder_contents(save_folder + "masks" + os.sep)
        else:
            os.makedirs(save_folder + "masks" + os.sep)

    # iterate over all files in folder and subfolders
    for file in os.listdir(folder_name):
        if os.path.isdir(folder_name + file):
            get_all_images_recurrently(folder_name + file + os.sep, recurrent_depth=recurrent_depth + 1, relevant_masks=relevant_masks, save_folder=save_folder)
        elif file.endswith('.jpg'):
            # print out 10% of files just to know that the program is running
            if random.randint(0, 100) < 10:
                print(file)

            # find corresponding folder for masks
            mask_folder = folder_name + file.replace('.jpg', '') + os.sep
            image_json = load_json(folder_name + file.replace('.jpg', '.json'))
            mask_ids_pairs = _get_relevant_json_data(image_json)

            # check if there are any masks of interest
            total_len = 0
            present_masks = []
            for mask in relevant_masks:
                total_len += len(mask_ids_pairs[mask])
                if len(mask_ids_pairs[mask]) > 0:
                    present_masks.append(mask)
            if total_len == 0:
                print("skipping", folder_name + file)
                continue

            # load found masks
            mask_multiple_masks_pairs = {}
            for mask in present_masks:
                masks = []
                for id in mask_ids_pairs[mask]:
                    mask_file_id = mask_folder + _get_mask_file_name(mask_folder, id)
                    mask_array = cv2.imread(mask_file_id, cv2.IMREAD_GRAYSCALE)
                    mask_array = cv2.resize(mask_array, (seg_cf.WIDTH, seg_cf.HEIGHT), interpolation=cv2.INTER_NEAREST)
                    mask_array[mask_array > 200] = seg_cf.MASKS[mask]
                    mask_array[mask_array != seg_cf.MASKS[mask]] = 0
                    masks.append(mask_array)

                # skip if no masks of this class were found
                if len(masks) == 0:
                    continue

                # combine masks of same class
                combined_mask = None
                for mask_i in masks:
                    if combined_mask is None:
                        combined_mask = mask_i
                    else:
                        combined_mask = np.add(combined_mask, mask_i)
                # make sure that there are no overlapping masks
                combined_mask[combined_mask > max(seg_cf.MASKS.values())] = 0

                mask_multiple_masks_pairs[mask] = combined_mask.copy()

            # combine all masks of different classes
            combined_mask_multi_class = None
            for mask in mask_multiple_masks_pairs:
                if combined_mask_multi_class is None:
                    combined_mask_multi_class = mask_multiple_masks_pairs[mask]
                else:
                    combined_mask_multi_class = np.add(combined_mask_multi_class, mask_multiple_masks_pairs[mask])

            # make sure that there are no overlapping masks. Overlapping masks will be greater than the maximum mask index, given there are only 2 masks
            combined_mask_multi_class[combined_mask_multi_class > max(seg_cf.MASKS.values())] = 0

            # TEMPORARILY MAKE MASKS VISIBLE
            # combined_mask *= 100

            # save image and mask in acceptable format
            image_save_file = save_folder + "images" + os.sep + file
            mask_save_file = save_folder + "masks" + os.sep + file.replace('.jpg', '.png')

            # resize image and save it to save folder
            img = cv2.imread(folder_name + file)
            img = cv2.resize(img, (seg_cf.HEIGHT, seg_cf.WIDTH))
            cv2.imwrite(image_save_file, img)

            # resize mask and save mask to save folder
            cv2.imwrite(mask_save_file, combined_mask_multi_class)


train_image_paths = []
train_mask_paths = []
test_mask_paths = []
test_image_paths = []


def get_all_paths(folder_name, recurrent_depth=0):
    global train_image_paths, train_mask_paths, test_mask_paths, test_image_paths
    for file in os.listdir(folder_name):
        if os.path.isdir(folder_name + file):
            get_all_paths(folder_name + file + os.sep, recurrent_depth=recurrent_depth + 1)
        elif file.endswith('.jpg'):
            mask_path = folder_name.replace(os.sep + "images" + os.sep, os.sep + "masks" + os.sep) + file.replace('.jpg', '.png')
            if seg_cf.CHOSEN_MASK_TYPE == seg_cf.PIXEL_LEVEL_MASK_TYPE:
                if not seg_utils.check_presence_of_min_pixels_mask(mask_path):
                    continue

            if (os.sep + "train" + os.sep) in folder_name:
                train_image_paths.append(folder_name + file)
                train_mask_paths.append(mask_path)

                # make sure that the files exists
                assert os.path.exists(train_mask_paths[-1])
                assert os.path.exists(train_image_paths[-1])
            elif (os.sep + "test" + os.sep) in folder_name:
                test_image_paths.append(folder_name + file)
                test_mask_paths.append(mask_path)

                # make sure that the files exists
                assert os.path.exists(test_mask_paths[-1])
                assert os.path.exists(test_image_paths[-1])

    # save paths to file. We are doing this in recursion_depth == 0 because we want to save the paths only once
    if recurrent_depth == 0:
        print("train_image_paths:", len(train_image_paths))
        print("train_mask_paths:", len(train_mask_paths))
        print("test_image_paths:", len(test_image_paths))
        print("test_mask_paths:", len(test_mask_paths))

        # save paths to file
        if not os.path.exists(os.path.join(cf.TRAINING_FOLDER_PIXEL_LEVEL_PATHS[seg_cf.ADE20K], "paths_pkl")):
            os.makedirs(os.path.join(cf.TRAINING_FOLDER_PIXEL_LEVEL_PATHS[seg_cf.ADE20K], "paths_pkl"))

        with open(os.path.join(cf.TRAINING_FOLDER_PIXEL_LEVEL_PATHS[seg_cf.ADE20K], "paths_pkl", "paths.pkl"), "wb") as f:
            pickle.dump([train_image_paths, train_mask_paths, test_image_paths, test_mask_paths], f)

        # print random path/mask pairs to check if they are correct
        printed_indices_tr = []
        printed_indices_te = []
        for i in range(50):
            while True:
                ri_tr = random.randint(0, len(train_image_paths) - 1)
                if ri_tr not in printed_indices_tr:
                    printed_indices_tr.append(ri_tr)
                    break
            print(train_image_paths[ri_tr], train_mask_paths[ri_tr])

            while True:
                ri_te = random.randint(0, len(test_image_paths) - 1)
                if ri_te not in printed_indices_te:
                    printed_indices_te.append(ri_te)
                    break
            print(test_image_paths[ri_te], test_mask_paths[ri_te])
            print("_____________________________________________")


def get_borders_for_all_masks(target_folder=cf.TRAINING_FOLDER_PIXEL_LEVEL_PATHS[seg_cf.ADE20K],
                              save_folder_name=cf.TRAINING_FOLDER_BORDER_LEVEL_PATHS[seg_cf.ADE20K],
                              make_masks_visible=False,
                              resize_to=None):
    def get_border_and_combine(mask_path):
        mask_arr = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if resize_to is not None:
            mask_arr = cv2.resize(mask_arr, resize_to, interpolation=cv2.INTER_NEAREST)

        border_pix_arr = seg_utils.make_border_of_mask(mask_arr,
                                                       combine_with_original_mask=True)

        return border_pix_arr

    # get all mask paths
    with open(target_folder + "paths_pkl" + os.sep + "paths.pkl", "rb") as f:
        train_image_paths, train_mask_paths, test_image_paths, test_mask_paths = pickle.load(f)

    # delete train and test folder if they exist
    if os.path.exists(save_folder_name + "train" + os.sep):
        utils.delete_folder_contents(save_folder_name + "train" + os.sep)
    if os.path.exists(save_folder_name + "test" + os.sep):
        utils.delete_folder_contents(save_folder_name + "test" + os.sep)

    # get all masks and images for train
    train_border_mask_paths = []
    train_border_image_paths = []
    for i, mask_path in enumerate(train_mask_paths):
        print("Train: ", i, "/", len(train_mask_paths), "cnt:", len(train_border_mask_paths), mask_path.split(os.sep)[-1])
        # create and move mask to new folder
        mask_save_path = os.path.join(save_folder_name, "train", "masks", mask_path.split(os.sep)[-1])
        save_folder_path = os.path.dirname(mask_save_path)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        # get border
        border_pix_arr = get_border_and_combine(mask_path)

        # check if mask is valid
        if not seg_utils.check_presence_of_min_pixels_mask(border_pix_arr, chosen_classes=[seg_cf.PIXEL_LEVEL], min_pixels_mask_all_classes_per=seg_cf.MIN_PIXELS_MASK_ALL_CLASSES_PER):
            continue

        # make mask visible
        if make_masks_visible:
            border_pix_arr *= 80

        # will be saved if mask is valid
        train_border_mask_paths.append(mask_save_path)

        # save mask
        cv2.imwrite(mask_save_path, border_pix_arr)

        # create and move image to new folder
        image_path = train_image_paths[i]
        image_save_path = os.path.join(save_folder_name, "train", "images", image_path.split(os.sep)[-1])
        save_folder_path = os.path.dirname(image_save_path)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        train_border_image_paths.append(image_save_path)
        if resize_to is None:
            shutil.copy(image_path, image_save_path)
        else:
            image = cv2.imread(image_path)
            image = cv2.resize(image, resize_to, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(image_save_path, image)

    # get all masks and images for test
    test_border_mask_paths = []
    test_border_image_paths = []
    for i, mask_path in enumerate(test_mask_paths):
        print("Test: ", i, "/", len(test_mask_paths), "cnt:", len(test_border_mask_paths), mask_path.split(os.sep)[-1])
        # create and move mask to new folder
        mask_save_path = os.path.join(save_folder_name, "test", "masks", mask_path.split(os.sep)[-1])
        save_folder_path = os.path.dirname(mask_save_path)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        # get border
        border_pix_arr = get_border_and_combine(mask_path)

        # check if mask is valid
        if not seg_utils.check_presence_of_min_pixels_mask(border_pix_arr, chosen_classes=[seg_cf.PIXEL_LEVEL], min_pixels_mask_all_classes_per=seg_cf.MIN_PIXELS_MASK_ALL_CLASSES_PER):
            continue

        # make mask visible
        if make_masks_visible:
            border_pix_arr *= 80

        test_border_mask_paths.append(mask_save_path)

        # save mask
        cv2.imwrite(mask_save_path, border_pix_arr)

        # create and move image to new folder
        image_path = test_image_paths[i]
        image_save_path = os.path.join(save_folder_name, "test", "images", image_path.split(os.sep)[-1])
        save_folder_path = os.path.dirname(image_save_path)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        test_border_image_paths.append(image_save_path)
        if resize_to is None:
            shutil.copy(image_path, image_save_path)
        else:
            image = cv2.imread(image_path)
            image = cv2.resize(image, resize_to, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(image_save_path, image)

    # save border paths in a pickle file
    if not os.path.exists(save_folder_name + "paths_pkl"):
        os.makedirs(save_folder_name + "paths_pkl")
    with open(save_folder_name + "paths_pkl" + os.sep + "paths.pkl", "wb") as f:
        pickle.dump([train_border_image_paths, train_border_mask_paths, test_border_image_paths, test_border_mask_paths], f)


if __name__ == '__main__':
    # below must be run ONLY with CHOSEN_MASK_TYPE == PIXEL_LEVEL_MASK_TYPE
    # get_all_images_recurrently(os.path.join(cf.ORG_DATASET_FOLDERS[seg_cf.ADE20K], "images", "ADE", "training") + os.sep,
    #                            save_folder=cf.TRAINING_FOLDER_PIXEL_LEVEL_PATHS[seg_cf.ADE20K] + "train" + os.sep)
    # get_all_images_recurrently(os.path.join(cf.ORG_DATASET_FOLDERS[seg_cf.ADE20K], "images", "ADE", "validation") + os.sep,
    #                            save_folder=cf.TRAINING_FOLDER_PIXEL_LEVEL_PATHS[seg_cf.ADE20K] + "test" + os.sep)

    ####################################################################################################################
    # below must be run with the CHOSEN_MASK_TYPE of your choice so that pixel filter is not activated here when not using pixel filter. Only needs to be run once to save the paths in a pickle file
    get_all_paths(cf.TRAINING_FOLDER_PIXEL_LEVEL_PATHS[seg_cf.ADE20K])

    # assert seg_cf.CHOSEN_MASK_TYPE == seg_cf.BORDER_MASK_TYPE, "Must be run with CHOSEN_MASK_TYPE == BORDER_MASK_TYPE"
    # below must be run ONLY with CHOSEN_MASK_TYPE == BORDER_MASK_TYPE
    # get_borders_for_all_masks(make_masks_visible=False)
    ####################################################################################################################
