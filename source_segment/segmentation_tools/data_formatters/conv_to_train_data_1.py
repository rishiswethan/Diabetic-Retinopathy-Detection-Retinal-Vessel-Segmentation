import os
import cv2
import pickle
import random
import csv

import source_segment.segmentation_tools.utils as seg_utils

# CHASEDB, DRIVE, STARE, HRF, DR-HAGIS
# ds_name = "CHASEDB"
# ds_name = "DRIVE"
# ds_name = "STARE"
# ds_name = "HRF"
ds_name = "DR-HAGIS"

images_path = f"/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/FundusImages/data_segment/org_data/{ds_name}/images/"
masks_path = f"/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/FundusImages/data_segment/org_data/{ds_name}/manual/"
test_all_csv = f"/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/FundusImages/data_segment/org_data/{ds_name}/test_all.csv"
extracted_training_folder = f"/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/FundusImages/data_segment/training_data/{ds_name}/"
split_ratio = 1.0
make_mask_binary = True

############################################################################################################################################

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
    print(mask, mask_img_paths[mask])

    # read image and mask and save them. just an easy way to convert to png and jpg
    if mask_img_paths[mask].endswith(".gif"):
        cap = cv2.VideoCapture(image_file_path)
        ret, image = cap.read()
        cap.release()
    else:
        image = cv2.imread(image_file_path)

    if mask.endswith(".gif"):
        cap = cv2.VideoCapture(mask_file_path)
        ret, mask = cap.read()
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        print("mask:", mask.shape)
        cap.release()
    else:
        mask = cv2.imread(mask_file_path)

    if mask is None or image is None:
        print("Error reading image or mask")
        print("image:", image)
        print("mask:", mask)
        continue

    if make_mask_binary:
        mask[mask > 0] = 1

    # split into train and test randomly
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
