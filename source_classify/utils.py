import os
import shutil
import warnings
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

import source_classify.config as cf

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


def get_min_max_value(history, key, get_index=False):
    min = 99999
    max = -99999
    min_index = 0
    max_index = 0
    for i in range(len(history)):
        if history[i][key] < min:
            min = history[i][key]
            min_index = i

        if history[i][key] > max:
            max = history[i][key]
            max_index = i

    if get_index:
        return min, max, min_index, max_index

    return min, max


def plot_history(history, save_path):
    """
    Plots the training and validation losses and accuracies.
    """

    losses = [x['train_loss'] for x in history]
    val_losses = [x['val_loss'] for x in history]
    accs = [x['train_acc'] for x in history]
    val_accs = [x['val_acc'] for x in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses, '-x', label='train_loss')
    ax1.plot(val_losses, '-x', label='val_loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.legend()
    ax1.set_title('Loss vs. No. of epochs')

    ax2.plot(accs, '-x', label='train_acc')
    ax2.plot(val_accs, '-x', label='val_acc')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    ax2.legend()
    ax2.set_title('Accuracy vs. No. of epochs')

    plt.savefig(save_path)



def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plots the confusion matrix. Set parameter `cm` to the confusion matrix and
    `class_names` to the names of the classes.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    cm = np.sum(cm, axis=0).reshape(len(class_names), len(class_names))  # Summing up all confusion matrices and reshaping to a 2D array
    sns.heatmap(cm, annot=True, fmt='.2f',
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues', ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(save_path)

