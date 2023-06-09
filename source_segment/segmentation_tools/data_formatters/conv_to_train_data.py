import os
import cv2
import pickle
import random
import csv

import source_segment.segmentation_tools.utils as seg_utils


images_path = "/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/FundusImages/data_segment/org_data/STARE/images/"
masks_path = "/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/FundusImages/data_segment/org_data/STARE/manual/"
test_all_csv = "/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/FundusImages/data_segment/org_data/STARE/test_all.csv"
extracted_training_folder = "/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/FundusImages/data_segment/training_data/STARE/"
split_ratio = 1.0

ext_imgs_folder = extracted_training_folder + "images" + os.sep
ext_masks_folder = extracted_training_folder + "masks" + os.sep

# create folders if they don't exist
seg_utils.clear_folder(extracted_training_folder + "train" + os.sep + "images" + os.sep)
seg_utils.clear_folder(extracted_training_folder + "train" + os.sep + "masks" + os.sep)
seg_utils.clear_folder(extracted_training_folder + "test" + os.sep + "images" + os.sep)
seg_utils.clear_folder(extracted_training_folder + "test" + os.sep + "masks" + os.sep)

# iterate through masks folder and find the corresponding image
images_paths = os.listdir(images_path)
masks_paths = os.listdir(masks_path)

# read csv file and create dictionary of im_paths and gt_paths
with open(test_all_csv, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    mask_img_paths = {}
    for i, row in enumerate(reader):
        if i == 0:
            continue
        mask_img_paths[row[1].split('/')[-1]] = row[0].split('/')[-1]

print(mask_img_paths)

# create dictionary of images without extension
images_no_ext = {}
for file in images_paths:
    images_no_ext[file.split(".")[0]] = file

# iterate through masks and find corresponding image, then copy both to train or test folder
for mask in masks_paths:
    if not mask_img_paths.get(mask):
        continue

    # full image and mask paths
    mask_file_path = masks_path + mask
    image_file_path = images_path + mask_img_paths[mask]

    # target image and mask paths
    image_file_tgt = mask.split(".")[0] + ".jpg"
    mask_file_tgt = mask.split(".")[0] + ".png"
    print(image_file_tgt, mask_file_tgt)

    image = cv2.imread(image_file_path)
    mask = cv2.imread(mask_file_path)

    if random.random() < split_ratio:
        ext_imgs_folder_train_test = ext_imgs_folder.replace(f"images{os.sep}", f"train{os.sep}images{os.sep}")
        ext_masks_folder_train_test = ext_masks_folder.replace(f"masks{os.sep}", f"train{os.sep}masks{os.sep}")
    else:
        ext_imgs_folder_train_test = ext_imgs_folder.replace(f"images{os.sep}", f"test{os.sep}images{os.sep}")
        ext_masks_folder_train_test = ext_masks_folder.replace(f"masks{os.sep}", f"test{os.sep}masks{os.sep}")

    # save image
    cv2.imwrite(ext_imgs_folder_train_test + image_file_tgt, image)
    # save mask
    cv2.imwrite(ext_masks_folder_train_test + mask_file_tgt, mask)
