import os
import cv2
import pickle
import random

import source_segment.segmentation_tools.utils as seg_utils
import source_segment.segmentation_tools.segmentation_config as seg_cf


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