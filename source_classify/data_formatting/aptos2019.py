import os
import random
import shutil
import csv

import source.utils as utils
import source.config as config


def correct_file_names(original_all_folders_path, training_all_folder_path):
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
    train_images_path = original_all_folders_path + "train_images" + os.sep
    train_csv_path = original_all_folders_path + "train.csv"

    train_images = os.listdir(train_images_path)
    train_images = [train_images_path + image for image in train_images]

    # get the labels for each image and skip the first row (header)
    unique_labels = []
    train_labels = {}
    with open(train_csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for i, row in enumerate(csv_reader):
            train_labels[row["id_code"]] = row["diagnosis"]
            if FULL_LABELS[int(row["diagnosis"])] not in unique_labels:
                unique_labels.append(FULL_LABELS[int(row["diagnosis"])])

    # prepare the training folder and create the various subfolders
    utils.clear_folder(training_all_folder_path, create_if_not_exists=True)
    os.makedirs(training_all_folder_path + "all" + os.sep)
    for folder_name in unique_labels:
        # get label number from the folder name
        folder_number = str(list(FULL_LABELS.keys())[list(FULL_LABELS.values()).index(folder_name)])

        os.makedirs(training_all_folder_path + "all" + os.sep + folder_name + "-" + folder_number)

    # copy the files from the original folder to the training folder and rename them to contain the folder name
    for image in train_images:
        folder_name = train_labels[image.split(os.sep)[-1].split(".")[0]]
        folder_name = FULL_LABELS[int(folder_name)] + "-" + folder_name

        rename = image.split(os.sep)[-1].split(".")[0] + "-" + folder_name + "." + image.split(os.sep)[-1].split(".")[1]
        shutil.copy(image, training_all_folder_path + "all" + os.sep + folder_name + os.sep + rename)


def split_train_test(all_folders_path, split_per):
    """
    Split the folders into train and test sets. Max images in each folder will be the minimum of the split_per
    in the all_folders_path list.
    """

    # get a list of all the folders and the number of images in each folder
    all_folders = []
    all_folders_images = []
    print("\n\nAll folders path: ", all_folders_path)
    for folder in os.listdir(all_folders_path):
        if not os.path.isdir(all_folders_path + folder):
            continue

        print("Folder: ", folder)
        all_folders.append(folder)
        all_folders_images.append(len(os.listdir(all_folders_path + folder)))

    # find the minimum number of images in each folder and set the max_test_images to (x * split_per)
    min_images = min(all_folders_images)
    max_test_images = int(min_images * split_per)
    print("Total images: ", sum(all_folders_images))
    print("Max test images per folder: ", max_test_images)
    print("total train images: ", sum(all_folders_images) - (len(all_folders) * max_test_images))
    print("total test images: ", len(all_folders) * max_test_images)

    # create the train and test folders
    parent_folder = all_folders_path.split(os.sep)[-2]
    parent_folder = all_folders_path.split(parent_folder + os.sep)[0]

    # prepare the train and test folders
    train_folder = os.path.dirname(parent_folder) + os.sep + 'train' + os.sep
    test_folder = os.path.dirname(parent_folder) + os.sep + 'test' + os.sep
    print("Train folder: ", train_folder)
    print("Test folder: ", test_folder)
    utils.clear_folder(train_folder, create_if_not_exists=True)
    utils.clear_folder(test_folder, create_if_not_exists=True)

    # loop through all the folders, shuffle the images, and move the images to the train and test folders
    for folder in all_folders:
        all_images = os.listdir(all_folders_path + folder)
        random.shuffle(all_images)
        for image in all_images[:max_test_images]:
            shutil.copy(
                all_folders_path + folder + os.sep + image,
                test_folder + os.sep + image
            )
        for image in all_images[max_test_images:]:
            shutil.copy(
                all_folders_path + folder + os.sep + image,
                train_folder + os.sep + image
            )


if __name__ == '__main__':
    correct_file_names(
        original_all_folders_path=config.DATA_FOLDERS['org_data'] + config.APTOS2019 + os.sep,
        training_all_folder_path=config.DATA_FOLDERS['training_data'] + config.APTOS2019 + os.sep
    )

    split_train_test(
        config.DATA_FOLDERS['training_data'] + config.APTOS2019 + os.sep + 'all' + os.sep,
        0.3
    )

