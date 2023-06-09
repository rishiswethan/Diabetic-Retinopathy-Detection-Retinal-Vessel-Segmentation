from PIL import Image
import numpy as np
import os
import random
import pickle
import threading

import source_segment.config as cf
import source_segment.segmentation_tools.segmentation_config as seg_cf
import source_segment.utils as utils
import source_segment.segmentation_tools.utils as seg_utils

train_images = []
train_masks = []
test_images = []
test_masks = []


# wall is 12
# floor is 5
def remove_all_masks_except(mask_path, relevant_mask_numbers, replacement_numbers, save_path):
    mask = np.array(Image.open(mask_path))
    for i, mask_number in enumerate(relevant_mask_numbers):
        mask[mask == mask_number] = replacement_numbers[i]
    mask[mask > max(replacement_numbers)] = 0
    # Image.fromarray(mask * 100).show()
    Image.fromarray(mask).save(save_path)


# recurrently call remove all masks on given folder and save it to another folder
def call_remove_all_masks_recursively(folder_path, relevant_mask_numbers, replacement_numbers, save_folder_name_replacement=['labels_13', 'masks'], recursion_depth=0):
    if recursion_depth <= 3:
        print(folder_path.split(cf.ORG_DATASET_FOLDERS[seg_cf.SCENE_NET])[1])
    for file in os.listdir(folder_path):
        if os.path.isdir(folder_path + file):
            call_remove_all_masks_recursively(folder_path + file + os.sep,
                                              relevant_mask_numbers=relevant_mask_numbers,
                                              replacement_numbers=replacement_numbers,
                                              save_folder_name_replacement=save_folder_name_replacement,
                                              recursion_depth=recursion_depth + 1)
        elif file.endswith('.png'):
            # saving masks in a separate folder that mimics the original folder structure
            save_folder_name_new = folder_path.replace(save_folder_name_replacement[0], save_folder_name_replacement[1])
            # create folder(s) if it does not exist
            if not os.path.exists(save_folder_name_new):
                os.makedirs(save_folder_name_new)
            # remove file if it already exists
            if os.path.exists(save_folder_name_new + file):
                os.remove(save_folder_name_new + file)

            # print(folder_path.split(cf.ORG_DATASET_FOLDERS[seg_cf.SCENE_NET])[1] + file, " to", save_folder_name_new.split(cf.ORG_DATASET_FOLDERS[seg_cf.SCENE_NET])[1] + file, "...")
            remove_all_masks_except(mask_path=folder_path + file,
                                    relevant_mask_numbers=relevant_mask_numbers,
                                    replacement_numbers=replacement_numbers,
                                    save_path=save_folder_name_new + file)


def _get_corresponding_mask_path(train_image_path):
    return train_image_path.replace(os.sep + 'images' + os.sep, os.sep + 'masks' + os.sep).replace('.jpg', '.png')


def get_all_paths(folder_path, recursion_depth=0):
    global train_images, train_masks, test_images, test_masks
    for file in os.listdir(folder_path):
        # skip masks folder. We are adding masks by replacing the path string
        if (os.sep + 'masks' + os.sep) in folder_path:
            continue

        if recursion_depth <= 2:
            print(str(folder_path + file).replace(cf.TRAINING_FOLDER_PATHS[seg_cf.SCENE_NET], ''))
            print("train:", len(train_images))
            print("test:", len(test_images))

        if os.path.isdir(folder_path + file + os.sep):
            # call function recursively until we reach an image
            get_all_paths(folder_path + file + os.sep, recursion_depth + 1)
        elif file.endswith('.jpg'):
            # find if we are in train or test folder, get the corresponding mask path and add it to the list accordingly
            if (os.sep + 'train' + os.sep) in folder_path:
                train_images.append(folder_path + file)
                train_masks.append(_get_corresponding_mask_path(folder_path + file))

                # make sure the files exist
                assert os.path.exists(train_masks[-1]), train_masks[-1]
                assert os.path.exists(train_images[-1]), train_images[-1]

            elif (os.sep + 'test' + os.sep) in folder_path:
                test_images.append(folder_path + file)
                test_masks.append(_get_corresponding_mask_path(folder_path + file))

                # make sure the files exist
                assert os.path.exists(test_masks[-1]), test_masks[-1]
                assert os.path.exists(test_images[-1]), test_images[-1]

    # save paths to file. We are doing this in recursion_depth == 0 because we want to save the paths only once
    if recursion_depth == 0:
        print("\n\n\n_____________________________________________")
        print("train_images:", len(train_images))
        print("train_masks:", len(train_masks))
        print("test_images:", len(test_images))
        print("test_masks:", len(test_masks))

        # shuffle lists
        train_images = seg_utils.shuffle_train_data(train_images)
        train_masks = seg_utils.shuffle_train_data(train_masks)
        test_images = seg_utils.shuffle_train_data(test_images)
        test_masks = seg_utils.shuffle_train_data(test_masks)

        # save paths to file
        with open(os.path.join(cf.TRAINING_FOLDER_PATHS[seg_cf.SCENE_NET], "paths_pkl", "paths.pkl"), "wb") as f:
            pickle.dump([train_images, train_masks, test_images, test_masks], f)

        # print random path/mask pairs to check if they are correct
        printed_indices_tr = []
        printed_indices_te = []
        for i in range(50):
            while True:
                ri_tr = random.randint(0, len(train_images) - 1)
                if ri_tr not in printed_indices_tr:
                    printed_indices_tr.append(ri_tr)
                    break
            print(train_images[ri_tr], train_masks[ri_tr])

            while True:
                ri_te = random.randint(0, len(test_images) - 1)
                if ri_te not in printed_indices_te:
                    printed_indices_te.append(ri_te)
                    break
            print(test_images[ri_te], test_masks[ri_te])
            print("_____________________________________________")


train_image_folders = []
train_mask_folders = []
test_image_folders = []
test_mask_folders = []
image_names = []
mask_names = []


def get_unique_folders(folder_name, recursion_depth=0):
    global train_image_folders, train_mask_folders, test_image_folders, test_mask_folders, image_names, mask_names
    if recursion_depth == 0:
        # clear lists
        train_image_folders = []
        train_mask_folders = []
        test_image_folders = []
        test_mask_folders = []
        image_names = []
        mask_names = []

    for file in os.listdir(folder_name):
        # we don't need to check masks folder name
        if (os.sep + "masks" + os.sep) in folder_name:
            continue

        if os.path.isdir(folder_name + file):
            get_unique_folders(folder_name + file + os.sep, recursion_depth + 1)
        else:
            # We have reached a file. This would be a folder that would indicate a single room
            if len(image_names) == 0:
                # get list of images only one time
                for sub_file in os.listdir(folder_name):
                    if (os.sep + "images" + os.sep) in folder_name:
                        image_names.append(sub_file)
                        mask_names.append(sub_file.replace('.jpg', '.png'))
            # store the folder names
            if (os.sep + "train" + os.sep) in folder_name:
                train_image_folders.append(folder_name)
                train_mask_folders.append(folder_name.replace(os.sep + "images" + os.sep, os.sep + "masks" + os.sep))
            elif (os.sep + "test" + os.sep) in folder_name:
                test_image_folders.append(folder_name)
                test_mask_folders.append(folder_name.replace(os.sep + "images" + os.sep, os.sep + "masks" + os.sep))

            # break the loop because we don't need to check the rest of the files in the folder
            break

    # return the list if we are in the root folder
    if recursion_depth == 0:
        # print("train_image_folders:", len(train_image_folders))
        # print("train_mask_folders:", len(train_mask_folders))
        # print("test_image_folders:", len(test_image_folders))
        # print("test_mask_folders:", len(test_mask_folders))
        # print("image_names:", len(image_names))
        # print("mask_names:", len(mask_names))

        return train_image_folders, train_mask_folders, test_image_folders, test_mask_folders, image_names, mask_names


if __name__ == '__main__':
    # call_remove_all_masks_recursively(cf.ORG_DATASET_FOLDERS[seg_cf.SCENE_NET] + 'train' + os.sep + 'labels_13' + os.sep,
    #                                   relevant_mask_numbers=[12, 5],
    #                                   replacement_numbers=[seg_cf.MASKS[seg_cf.WALL], seg_cf.MASKS[seg_cf.FLOOR]])

    # for i in range(0, 17):
    #     # call the function in thread to speed up the process
    #     t = threading.Thread(target=call_remove_all_masks_recursively,
    #                          args=(
    #                              cf.ORG_DATASET_FOLDERS[seg_cf.SCENE_NET] + 'train' + os.sep + 'labels_13' + os.sep + str(i) + os.sep,
    #                              [12, 5],
    #                              [seg_cf.MASKS[seg_cf.WALL], seg_cf.MASKS[seg_cf.FLOOR]]
    #                          ))
    #     t.start()

    # call_remove_all_masks_recursively(
    #     cf.ORG_DATASET_FOLDERS[seg_cf.SCENE_NET] + 'valid' + os.sep + 'labels_13' + os.sep + '0' + os.sep,
    #     [12, 5],
    #     [seg_cf.MASKS[seg_cf.WALL], seg_cf.MASKS[seg_cf.FLOOR]]
    # )

    # get_all_paths(cf.TRAINING_FOLDER_PATHS[seg_cf.SCENE_NET])
    get_unique_folders(cf.TRAINING_FOLDER_PATHS[seg_cf.SCENE_NET])
