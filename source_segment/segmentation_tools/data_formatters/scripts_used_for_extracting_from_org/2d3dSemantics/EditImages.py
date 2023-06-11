import os
import sys

import numpy as np
import imageio.v2
import json
import cv2


def parse_label(label):
    """ Parses a label into a dict """
    res = {}
    clazz, instance_num, room_type, room_num, area_num = label.split("_")
    res['instance_class'] = clazz
    res['instance_num'] = int(instance_num)
    res['room_type'] = room_type
    res['room_num'] = int(room_num)
    res['area_num'] = int(area_num)
    return res


def get_index(color):
    """ Parse a color as a base-256 number and returns the index
    Args:
        color: A 3-tuple in RGB-order where each element in [0, 255]
    Returns:
        index: an int containing the indec specified in 'color'
    """
    return color[0] * 256 * 256 + color[1] * 256 + color[2]


def load_labels(label_file):
    """ Convenience function for loading JSON labels """
    with open(label_file) as f:
        return json.load(f)


def get_mask(original_mask, all_labels, desired_mask, destination_directory, destination_file_name):
    """
    Filters the original mask by turning the desired mask white and the rest of the image black and saves it at:
    'destination_directory/destination_file_name'.
    :param original_mask: Semantic image in a numpy array
    :param all_labels: A list of all labels loaded from semantic_labels.json
    :param desired_mask: The mask we want to filter (eg: "wall")
    :param destination_directory: The directory we want to store the new image at (should NOT have '/' at the end).
    :param destination_file_name: Only the file name of the edited image (without its path)
    :return: none
    """
    desired_mask = str.lower(desired_mask)
    # print(len(all_labels))
    if not os.path.isdir(destination_directory):
        os.mkdir(destination_directory)
    new_mask = np.zeros((original_mask.shape[0], original_mask.shape[1]), dtype=np.uint8)
    for x in range(original_mask.shape[0]):
        for y in range(original_mask.shape[1]):
            current_pixel = original_mask[x][y]
            # print("x = ", x, ", y = ", y, ", current pixel = ", current_pixel)
            current_index = get_index(current_pixel)
            if current_index >= len(all_labels):
                continue
            current_label_raw = all_labels[current_index]
            current_label_parsed = parse_label(current_label_raw)
            # print(current_label_parsed)
            if str.lower(current_label_parsed['instance_class']) == desired_mask:
                new_mask[x][y] = 255
    full_file_name = destination_directory + "/" + destination_file_name
    cv2.imwrite(full_file_name, new_mask)


def get_all_masks(all_files, all_labels, desired_mask):
    """
    For all semantic images, filters the desired mask as white and rest of the image as black.
    :param all_files: The name of the file which contains the name of all files from which we wish to extract the masks.
    :param all_labels: A list of all labels extracted from semantic_labels.json.
    :param desired_mask: The mask we want to filter (eg: "wall")
    :return: None
    """
    file_names = open(all_files, "r")
    lines = file_names.readlines()
    images_edited = 0
    for line in lines:
        line = line.strip("\n")
        current_image = imageio.v2.imread(line)
        destination_directory = "data/" + desired_mask
        destination_file_name = line.split("/")[2]
        get_mask(current_image, all_labels, desired_mask, destination_directory, destination_file_name)
        images_edited += 1
        print("Number of images edited = ", images_edited)
    file_names.close()


get_all_masks("semantic_file_names.txt", load_labels("semantic_labels.json"), sys.argv[1])
