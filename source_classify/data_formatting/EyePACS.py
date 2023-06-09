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
            train_labels[row["image"]] = row["level"]
            if FULL_LABELS[int(row["level"])] not in unique_labels:
                unique_labels.append(FULL_LABELS[int(row["level"])])

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
        if random.random() < 0.02:
            print(image.split(os.sep)[-1].split(".")[0])

        folder_name = train_labels[image.split(os.sep)[-1].split(".")[0]]
        folder_name = FULL_LABELS[int(folder_name)] + "-" + folder_name

        rename = image.split(os.sep)[-1].split(".")[0] + "-" + folder_name + "." + image.split(os.sep)[-1].split(".")[1]
        rename = img_name_prefix + rename
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
    # copy the first max_test_images to the train folder. Make sure to copy both the left and right images
    for folder in all_folders:
        all_images = os.listdir(all_folders_path + folder)

        # find unique images
        unique_images = []
        for image in all_images:
            if image.split("_")[0] not in unique_images:
                unique_images.append(image.split("_")[0])

        random.shuffle(unique_images)
        copied_image_names = []

        # copy the first max_test_images to the test folder. Make sure to copy both the left and right images
        for image in unique_images[max_test_images:]:
            # if the image has already been copied, then skip it
            if image in copied_image_names:
                continue

            print("Image: ", image)

            # get the left and right image names
            right_image = image + f"_right-{folder}.jpeg"
            left_image = image + f"_left-{folder}.jpeg"

            # copy the right image
            if os.path.exists(all_folders_path + folder + os.sep + right_image):
                shutil.copy(
                    all_folders_path + folder + os.sep + right_image,
                    train_folder + os.sep + right_image
                )
                copied_image_names.append(image)

            # copy the left image
            if os.path.exists(all_folders_path + folder + os.sep + left_image):
                shutil.copy(
                    all_folders_path + folder + os.sep + left_image,
                    train_folder + os.sep + left_image
                )
                copied_image_names.append(image)

        # copy the remaining to the test folder. Make sure to copy both the left and right images
        for image in unique_images[:max_test_images]:
            # if the image has already been copied, then skip it
            if image in copied_image_names:
                print("Image already copied: ", image)
                continue

            print("Image: ", image)

            # get the left and right image names
            right_image = image + f"_right-{folder}.jpeg"
            left_image = image + f"_left-{folder}.jpeg"

            # copy the right image
            if os.path.exists(all_folders_path + folder + os.sep + right_image):
                shutil.copy(
                    all_folders_path + folder + os.sep + right_image,
                    test_folder + os.sep + right_image
                )
                copied_image_names.append(image)

            # copy the left image
            if os.path.exists(all_folders_path + folder + os.sep + left_image):
                shutil.copy(
                    all_folders_path + folder + os.sep + left_image,
                    test_folder + os.sep + left_image
                )
                copied_image_names.append(image)


def audit_train_and_test_images(train_images_folder, test_images_folder):
    """
    Audit the train and test images to make sure that there are no duplicates
    """

    # get a list of all the images in the train and test folders
    train_images = os.listdir(train_images_folder)
    train_images = [image.split("_")[0] + "-" + image.split("-")[-2].split(".")[0]
                    for image in train_images]

    test_images = os.listdir(test_images_folder)
    test_images = [image.split("_")[0] + "-" + image.split("-")[-2].split(".")[0]
                     for image in test_images]

    # find the duplicates
    duplicates = []
    for image in train_images:
        print("Image: ", image)
        if image in test_images:
            duplicates.append(image)

    print("Duplicates: ", duplicates)


if __name__ == "__main__":
    # train set of EyePACS

    # correct_file_names("/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/FundusImages/data/org_data/EyePACS/eyepacs_preprocess/eyepacs_preprocess/",
    #                    "/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/FundusImages/data/org_data/EyePACS/eyepacs_preprocess/trainLabels.csv",
    #                    config.DATA_FOLDERS["training_data"] + config.EyePACS + os.sep,
    #                    img_name_prefix="")

    # split_train_test(config.DATA_FOLDERS["training_data"] + config.EyePACS + os.sep + "all" + os.sep, 0.0)
    #
    # audit_train_and_test_images(config.DATA_FOLDERS["training_data"] + config.EyePACS + os.sep + "train" + os.sep,
    #                             config.DATA_FOLDERS["training_data"] + config.EyePACS + os.sep + "test" + os.sep)

    ################################################################################################################################################################
    # test set of EyePACS

    # correct_file_names("/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/FundusImages/data/org_data/EyePACS_test/test/",
    #                    "/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/FundusImages/data/org_data/EyePACS_test/retinopathy_solution.csv",
    #                    config.DATA_FOLDERS["training_data"] + config.EyePACS_test + os.sep,
    #                    img_name_prefix="")

    split_train_test(config.DATA_FOLDERS["training_data"] + config.EyePACS_test + os.sep + "all" + os.sep, 0.0)

    audit_train_and_test_images(config.DATA_FOLDERS["training_data"] + config.EyePACS_test + os.sep + "train" + os.sep,
                                config.DATA_FOLDERS["training_data"] + config.EyePACS_test + os.sep + "test" + os.sep)
