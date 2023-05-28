import os
import shutil
import warnings
import numpy as np
import json

import config as cf

NUM_CLASSES = cf.NUM_CLASSES


def clear_folder(folder_path, create_if_not_exists=True):
    """
    Clear a folder without deleting the folder itself

    :param folder_path: str, path to the folder
    :return: None
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder if it does not exist
        if create_if_not_exists:
            os.makedirs(folder_path)
        else:
            raise ValueError(f"Folder {folder_path} does not exist")

    # Iterate over all files and directories in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Use try/except to catch any errors while deleting
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # remove file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove directory
        except Exception as e:
            warnings.warn(f"Failed to delete {file_path}. Reason: {e}")


def get_one_hot_encoding(label, num_classes=NUM_CLASSES):
    """
    Get one hot encoding for a label
    """

    one_hot = [0] * num_classes
    one_hot[label] = 1
    return one_hot


def get_label_from_path(path, return_one_hot=True):
    label = path.split(os.sep)[-1]
    label = int(label.split('-')[-1].split('.')[0])

    if return_one_hot:
        label = get_one_hot_encoding(label)

    return label


def get_class_weights(classes, normalize=True):
    """
    Get class weights for a dictionary of classes

    Parameters
    ----------
    classes : dict
        Dictionary of classes
        Example: {'banana': 9000, 'lemon': 9000, 'mango': 4500}
        Output: [0.25 0.25 0.5]
    normalize : bool
        Normalize the class weights to sum to 1. Default: True

    Returns
    -------
    class_weights : list
    """

    # find the max number
    max_d = max(classes.values())

    # find how much x is each number from the max
    how_much_x_each_class = {}
    for c in classes:
        how_much_x_each_class[c] = max_d / classes[c]

    # divide each number by the max of all numbers
    class_weights = list(range(len(classes)))
    for i, c in enumerate(classes):
        class_weights[i] = how_much_x_each_class[c] / (sum(how_much_x_each_class.values()) / len(classes))

    if normalize:
        # normalize
        class_weights = np.array(class_weights) / np.sum(class_weights)

    return class_weights


def save_dict_as_json(file_name, dict, over_write=False):
    if os.path.exists(file_name) and over_write:
        with open(file_name) as f:
            existing_dict = json.load(f)

        existing_dict.update(dict)

        with open(file_name, 'w') as f:
            json.dump(existing_dict, f)
    else:
        with open(file_name, 'w') as f:
            json.dump(dict, f)


def load_dict_from_json(file_name):
    with open(file_name) as f:
        d = json.load(f)
        # print(d)

    return d
