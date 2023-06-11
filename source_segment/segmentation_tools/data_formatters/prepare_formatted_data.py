import os
import cv2
import pickle
import random

import source_segment.segmentation_tools.utils as seg_utils
import source_segment.segmentation_tools.segmentation_config as seg_cf
import source_segment.config as cf


train_image_paths = []
train_mask_paths = []
test_image_paths = []
test_mask_paths = []


def make_mask_binary(mask_path):
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask[mask > 0] = 1
    cv2.imwrite(mask_path, mask)


def get_all_paths(folder_name, recurrent_depth=0, dataset_mode=None):
    """
    :param folder_name:
    :param recurrent_depth:
    :param dataset_mode: Dataset mode can be either "train" or "test". If None, then all paths are returned
    :return:
    """

    global train_image_paths, train_mask_paths, test_mask_paths, test_image_paths

    for file in os.listdir(folder_name):
        if os.path.isdir(folder_name + file):
            get_all_paths(folder_name + file + os.sep, recurrent_depth=recurrent_depth + 1, dataset_mode=dataset_mode)
        elif file.endswith('.jpg'):
            mask_path = folder_name.replace(os.sep + "images" + os.sep, os.sep + "masks" + os.sep) + file.replace('.jpg', '.png')
            if seg_cf.CHOSEN_MASK_TYPE == seg_cf.PIXEL_LEVEL_MASK_TYPE:
                if not seg_utils.check_presence_of_min_pixels_mask(mask_path):
                    continue

            print("dataset_mode:", dataset_mode, dataset_mode is not None)
            if ((os.sep + "train" + os.sep) in folder_name) or (dataset_mode is not None):
                if dataset_mode == "test":
                    pass
                else:
                    train_image_paths.append(folder_name + file)
                    train_mask_paths.append(mask_path)

                    # make sure that the files exists
                    assert os.path.exists(train_mask_paths[-1])
                    assert os.path.exists(train_image_paths[-1])

            if ((os.sep + "test" + os.sep) in folder_name) or (dataset_mode is not None):
                if dataset_mode == "train":
                    pass
                else:
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
        if not os.path.exists(os.path.join(folder_name, "paths_pkl")):
            os.makedirs(os.path.join(folder_name, "paths_pkl"))

        with open(os.path.join(folder_name, "paths_pkl", "paths.pkl"), "wb") as f:
            pickle.dump([train_image_paths, train_mask_paths, test_image_paths, test_mask_paths], f)

        # print random path/mask pairs to check if they are correct
        printed_indices_tr = []
        printed_indices_te = []
        for i in range(10):
            try:
                for _ in range(len(train_image_paths)):
                    ri_tr = random.randint(0, len(train_image_paths) - 1)
                    if ri_tr not in printed_indices_tr:
                        printed_indices_tr.append(ri_tr)
                        break
                print(train_image_paths[ri_tr], train_mask_paths[ri_tr])
            except ValueError:
                print("No train images found")
            except UnboundLocalError:
                pass
            except:
                raise

            try:
                for _ in range(len(test_image_paths)):
                    ri_te = random.randint(0, len(test_image_paths) - 1)
                    if ri_te not in printed_indices_te:
                        printed_indices_te.append(ri_te)
                        break
                print(test_image_paths[ri_te], test_mask_paths[ri_te])
            except ValueError:
                print("No test images found")
            except UnboundLocalError:
                pass
            except:
                raise
            print("_____________________________________________")


if __name__ == "__main__":
    # ds = ["CHASEDB", "train"]
    # ds = ["DR-HAGIS", "train"]
    # ds = ["DRIVE", "test"]
    # ds = ["HRF", "train"]
    # ds = ["STARE", "train"]
    ds = ["SMDG", "train"]

    get_all_paths(f"/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/FundusImages/data_segment/training_data/{ds[0]}/", dataset_mode=ds[1])
