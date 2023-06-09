import numpy as np
import os
import cv2
import shutil
import pickle
from PIL import Image

import source_segment.config as cf
import source_segment.segmentation_tools.segmentation_config as seg_cf
import source_segment.segmentation_tools.utils as seg_utils
import source_segment.utils as utils


def load_mask_as_binary_array(mask_path, binary_one_replacement=1, binary_th=230):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask[mask < binary_th] = 0
    mask[mask != 0] = binary_one_replacement

    return mask


cnt = 0
file_names = []


def get_images_masks_recurrently(folder_name, save_path, recurrent_depth=0):
    global cnt, file_names

    if recurrent_depth == 0:
        if os.path.exists(save_path + "images" + os.sep):
            utils.delete_folder_contents(save_path + "images" + os.sep)
        else:
            os.makedirs(save_path + "images" + os.sep)

        if os.path.exists(save_path + "masks" + os.sep):
            utils.delete_folder_contents(save_path + "masks" + os.sep)
        else:
            os.makedirs(save_path + "masks" + os.sep)

    for file in os.listdir(folder_name):
        if os.path.isdir(folder_name + file) and (not file.startswith(".")):
            get_images_masks_recurrently(folder_name + file + os.sep, save_path, recurrent_depth + 1)
        elif file.endswith(".jpg") and (folder_name.endswith(os.sep + "image" + os.sep)):
            cnt += 1
            # check for multiple instances of the string "image" in the path
            assert folder_name.count(os.sep + "image" + os.sep) == 1, "Multiple folders named of 'image' in path. Script won't work: " + folder_name

            image_path = folder_name + file
            floor_mask_path = folder_name.replace(os.sep + "image" + os.sep, os.sep + "floor" + os.sep) + file.replace(".jpg", ".png")
            wall_mask_path = folder_name.replace(os.sep + "image" + os.sep, os.sep + "wall" + os.sep) + file.replace(".jpg", ".png")

            # check if the masks exist
            if not os.path.isfile(floor_mask_path):
                print("Floor mask not found:", floor_mask_path)
                continue
            if not os.path.isfile(wall_mask_path):
                print("Wall mask not found:", wall_mask_path)
                continue

            # get the masks
            floor_mask = load_mask_as_binary_array(floor_mask_path, binary_one_replacement=seg_cf.FLOOR_MASK["mask_number"])
            wall_mask = load_mask_as_binary_array(wall_mask_path, binary_one_replacement=seg_cf.WALL_MASK["mask_number"])

            # combine the masks
            combined_mask = np.add(floor_mask, wall_mask)

            # save the masks and images
            last_folder_name = folder_name.split(os.sep)[-3]
            save_file_name = "cnt" + str(cnt) + "+" + last_folder_name + "+" + file
            print(cnt, save_file_name)

            # shutil.copyfile(image_path, save_path + "images" + os.sep + save_file_name)
            img_arr = np.array(Image.open(image_path))
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            img_arr = cv2.resize(img_arr, (seg_cf.WIDTH, seg_cf.HEIGHT), interpolation=cv2.INTER_NEAREST)
            combined_mask = cv2.resize(combined_mask, (seg_cf.WIDTH, seg_cf.HEIGHT), interpolation=cv2.INTER_NEAREST)

            cv2.imwrite(save_path + "images" + os.sep + save_file_name, img_arr)
            cv2.imwrite(save_path + "masks" + os.sep + save_file_name.replace(".jpg", ".png"), combined_mask)

            file_names.append(save_path + "images" + os.sep + save_file_name)


def split_to_test_and_train(data_paths, test_path, train_path, save_target_folder, test_per=seg_cf.SUN_RGBD_TEST_SPLIT_PER):
    train_image_paths = []
    train_mask_paths = []
    test_image_paths = []
    test_mask_paths = []
    cnt = 0

    # delete the contents of train and test folders
    if os.path.exists(test_path + "images" + os.sep):
        utils.delete_folder_contents(test_path + "images" + os.sep)
    else:
        os.makedirs(test_path + "images" + os.sep)

    if os.path.exists(test_path + "masks" + os.sep):
        utils.delete_folder_contents(test_path + "masks" + os.sep)
    else:
        os.makedirs(test_path + "masks" + os.sep)

    if os.path.exists(train_path + "images" + os.sep):
        utils.delete_folder_contents(train_path + "images" + os.sep)
    else:
        os.makedirs(train_path + "images" + os.sep)

    if os.path.exists(train_path + "masks" + os.sep):
        utils.delete_folder_contents(train_path + "masks" + os.sep)
    else:
        os.makedirs(train_path + "masks" + os.sep)

    for i, file in enumerate(data_paths):
        if os.path.isdir(file):
            continue

        print("Splitting: ", i, "/", len(data_paths), "cnt:", cnt)

        if i > len(data_paths) * (1 - test_per):
            mask_save_path = test_path + "masks" + os.sep + file.split(os.sep)[-1].replace(".jpg", ".png")
            current_mask_path = file.replace("images", "masks").replace(".jpg", ".png")

            # check if the mask has enough pixels only if pixel level mask is used
            if seg_cf.CHOSEN_MASK_TYPE == seg_cf.PIXEL_LEVEL_MASK_TYPE:
                if not seg_utils.check_presence_of_min_pixels_mask(current_mask_path):
                    continue

            shutil.copyfile(file,
                            test_path + "images" + os.sep + file.split(os.sep)[-1])
            shutil.copyfile(file.replace("images", "masks").replace(".jpg", ".png"),
                            mask_save_path)

            test_image_paths.append(test_path + "images" + os.sep + file.split(os.sep)[-1])
            test_mask_paths.append(test_path + "masks" + os.sep + file.split(os.sep)[-1].replace(".jpg", ".png"))
            cnt += 1
        else:
            mask_save_path = train_path + "masks" + os.sep + file.split(os.sep)[-1].replace(".jpg", ".png")
            current_mask_path = file.replace("images", "masks").replace(".jpg", ".png")

            # check if the mask has enough pixels only if pixel level mask is used
            if seg_cf.CHOSEN_MASK_TYPE == seg_cf.PIXEL_LEVEL_MASK_TYPE:
                if not seg_utils.check_presence_of_min_pixels_mask(current_mask_path):
                    continue

            shutil.copyfile(file,
                            train_path + "images" + os.sep + file.split(os.sep)[-1])
            shutil.copyfile(file.replace("images", "masks").replace(".jpg", ".png"),
                            mask_save_path)

            train_image_paths.append(train_path + "images" + os.sep + file.split(os.sep)[-1])
            train_mask_paths.append(train_path + "masks" + os.sep + file.split(os.sep)[-1].replace(".jpg", ".png"))
            cnt += 1

    if not os.path.exists(save_target_folder + "paths_pkl" + os.sep):
        os.makedirs(save_target_folder + "paths_pkl" + os.sep)
    # save paths in a pickle file
    with open(save_target_folder + "paths_pkl" + os.sep + "paths.pkl", "wb") as f:
        pickle.dump([train_image_paths, train_mask_paths, test_image_paths, test_mask_paths], f)

    print("Splitting complete")


def get_borders_for_all_masks(target_folder=cf.TRAINING_FOLDER_PIXEL_LEVEL_PATHS[seg_cf.SUN_RGBD],
                              save_folder_name=cf.TRAINING_FOLDER_BORDER_LEVEL_PATHS[seg_cf.SUN_RGBD],
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


if __name__ == "__main__":
    ######################################################################################################
    # Below will work in either of the CHOSEN_MASK_TYPE

    get_images_masks_recurrently(folder_name=cf.ORG_DATASET_FOLDERS[seg_cf.SUN_RGBD],
                                 save_path=cf.TRAINING_FOLDER_PIXEL_LEVEL_PATHS[seg_cf.SUN_RGBD] + "all" + os.sep)
    split_to_test_and_train(
        data_paths=file_names,
        test_path=cf.TRAINING_FOLDER_PIXEL_LEVEL_PATHS[seg_cf.SUN_RGBD] + "test" + os.sep,
        train_path=cf.TRAINING_FOLDER_PIXEL_LEVEL_PATHS[seg_cf.SUN_RGBD] + "train" + os.sep,
        save_target_folder=cf.TRAINING_FOLDER_PIXEL_LEVEL_PATHS[seg_cf.SUN_RGBD]
    )

    ######################################################################################################
    # Below will work only in CHOSEN_MASK_TYPE = "border"

    # assert seg_cf.CHOSEN_MASK_TYPE == seg_cf.BORDER_MASK_TYPE, "This will work only in CHOSEN_MASK_TYPE = BORDER_MASK_TYPE"
    # get_borders_for_all_masks(make_masks_visible=False)
