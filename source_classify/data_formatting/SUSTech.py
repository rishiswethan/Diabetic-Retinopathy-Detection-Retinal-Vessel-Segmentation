import source.config as config
import source.utils as utils
import source.data_formatting.aptos2019 as aptos2019

import os
import shutil
import csv
import random


def correct_file_names(original_all_folders_path, annot_csv, training_all_folder_path, clear_folder=True, img_name_prefix=""):
    """
    Make the file name <image_name>-<folder_name>.<extension> for all images in the original folder
    """

    FULL_LABELS = {
        0: 'No_DR',  # 1805 images
        1: 'Mild',  # 370 images
        2: 'Moderate',  # 999 images
        3: 'Severe',  # 193 images
        4: 'Proliferate_DR',  # 295 images
    }

    # get a list of all the folders and the number of images in each folder
    train_images_path = original_all_folders_path
    train_csv_path = annot_csv

    train_images = os.listdir(train_images_path)
    train_images = [train_images_path + image for image in train_images]

    # get the labels for each image and skip the first row (header)
    unique_labels = []
    train_labels = {}
    with open(train_csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for i, row in enumerate(csv_reader):
            train_labels[row["Fundus_images"]] = row["DR_grade_International_Clinical_DR_Severity_Scale"]
            if FULL_LABELS[int(row["DR_grade_International_Clinical_DR_Severity_Scale"])] not in unique_labels:
                unique_labels.append(FULL_LABELS[int(row["DR_grade_International_Clinical_DR_Severity_Scale"])])

    # prepare the training folder and create the various subfolders
    if clear_folder:
        utils.clear_folder(training_all_folder_path, create_if_not_exists=True)
        os.makedirs(training_all_folder_path + "all" + os.sep)

    for folder_name in unique_labels:
        # get label number from the folder name
        folder_number = str(list(FULL_LABELS.keys())[list(FULL_LABELS.values()).index(folder_name)])

        os.makedirs(training_all_folder_path + "all" + os.sep + folder_name + "-" + folder_number, exist_ok=True)

    # copy the files from the original folder to the training folder and rename them to contain the folder name
    for image in train_images:
        folder_name = train_labels[image.split(os.sep)[-1]]
        folder_name = FULL_LABELS[int(folder_name)] + "-" + folder_name

        rename = image.split(os.sep)[-1].split(".")[0] + "-" + folder_name + "." + image.split(os.sep)[-1].split(".")[1]
        rename = img_name_prefix + rename
        shutil.copy(image, training_all_folder_path + "all" + os.sep + folder_name + os.sep + rename)


if __name__ == "__main__":
    correct_file_names("/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/FundusImages/data/org_data/SUSTech_t/test/",
                       "/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/FundusImages/data/org_data/SUSTech/Labels.csv",
                       config.DATA_FOLDERS["training_data"] + config.SUSTech + os.sep,
                       img_name_prefix="")

    aptos2019.split_train_test(config.DATA_FOLDERS["training_data"] + config.SUSTech + os.sep + "all" + os.sep,
                               split_per=0.5)
