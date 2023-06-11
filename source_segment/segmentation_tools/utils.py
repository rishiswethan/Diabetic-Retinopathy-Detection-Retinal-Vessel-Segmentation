import h5py
import os
from sklearn.utils import shuffle
import shutil
import source_segment.config as cf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import warnings

import source_segment.segmentation_tools.segmentation_config as seg_cf


def save_h5(data, data_label, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset(data_label, data=data)


def split_to_test_and_train(data_path, test_path, train_path, test_per, seed=0):
    data_paths = os.listdir(data_path)
    print("before shuffle")
    print(data_paths[:50])
    data_paths = shuffle_train_data(data_paths, seed=seed)
    print("after shuffle")
    print(data_paths[:50])

    for i, file in enumerate(data_paths):
        if os.path.isdir(data_path + cf.ls + file):
            continue

        if i < len(data_paths) * test_per:
            print(f"Copying {file} to test")
            shutil.copyfile(
                (data_path + file),
                test_path + file
            )
        else:
            print(f"Copying {file} to train")
            shutil.copyfile(
                (data_path + file),
                train_path + file
            )
    print("Splitting complete")


def shuffle_train_data(array, seed=None):
    """
    Single array with be shuffled at axis 0
    Example:
        shuffle(X, seed=0)
        shuffle(Y, seed=0)

    Returns:
        Shuffled array.
        Sequence of shuffled copies of the collections. The original arrays
        are not impacted.
    """
    array = shuffle(array, random_state=seed)
    return array


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


################################################################################
# Class Weights
################################################################################
def get_dataset_counts(d):
    pixel_count = np.array([i for i in d.values()])

    sum_pixel_count = 0
    for i in pixel_count:
        sum_pixel_count += i

    return pixel_count, sum_pixel_count


def get_dataset_statistics(pixel_count, sum_pixel_count):
    pixel_frequency = np.round(pixel_count / sum_pixel_count, 4)

    mean_pixel_frequency = np.round(np.mean(pixel_frequency), 4)

    return pixel_frequency, mean_pixel_frequency


def get_balancing_class_weights(classes, d, disable_background=seg_cf.DISABLE_BACKGROUND_IN_METRICS, normalize=True):
    total_classes = list(seg_cf.MASKS.keys())
    pixel_count, sum_pixel_count = get_dataset_counts(d)

    if disable_background:
        # remove background from classes
        classes = [c for c in classes if c != seg_cf.UNLABELED]

    # combined pixel counts for unselected classes to background
    for c in total_classes:
        if (c not in classes) and (not disable_background):
            d[seg_cf.UNLABELED] += d[c]

    # find the max number
    max_d = max(d.values())

    # find how much x is each number from the max
    how_much_x_each_class = {}
    for c in classes:
        how_much_x_each_class[c] = max_d / d[c]

    # divide each number by the max of all numbers
    class_weights = list(range(len(classes)))
    for i, c in enumerate(classes):
        class_weights[i] = how_much_x_each_class[c] / (sum(how_much_x_each_class.values()) / len(classes))
        # class_weights[i] = how_much_x_each_class[c] / max(how_much_x_each_class.values())
        # class_weights[i] = how_much_x_each_class[c]

    if normalize:
        # normalize
        class_weights = np.array(class_weights) / np.sum(class_weights)

    return np.array(class_weights)


def get_dataset_pixel_counts(masks_path, chosen_masks=seg_cf.CHOSEN_MASKS, masks_dict=seg_cf.MASKS, survey_partially_per=1):
    pixel_count = {}
    # initialize pixel count dictionary
    for c in masks_dict:
        pixel_count[c] = 0

    if type(masks_path) == list:
        # if masks_path is a list of files
        for i, mask_file in enumerate(masks_path):
            # only continue every survey_partially_per. This is so we don't have to survey the entire dataset
            if np.random.rand() > survey_partially_per:
                continue
            if i % (len(masks_path) / 100) == 0:
                print(f"Surveyed {i} images")

            if check_presence_of_min_pixels_mask(mask_file):
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                for c in masks_dict:
                    pixel_count[c] += np.sum(mask == masks_dict[c], dtype=np.int64)
    else:
        # if masks_path is a directory
        for mask_file in os.listdir(masks_path):
            mask_file = os.path.join(masks_path, mask_file)
            if check_presence_of_min_pixels_mask(mask_file):
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                for c in masks_dict:
                    pixel_count[c] += np.sum(mask == masks_dict[c])

    print("Pixel count for each class: ", pixel_count)
    return pixel_count


def add_2_dicts(d1, d2):
    for key in d1:
        d1[key] += d2[key]

    return d1


def check_presence_of_min_pixels_mask(mask_path_or_arr,
                                      chosen_classes=seg_cf.CHECK_MIN_PIXELS_CLASSES,
                                      masks_dict=seg_cf.MASKS,
                                      min_pixels_mask_all_classes_per=seg_cf.MIN_PIXELS_MASK_ALL_CLASSES_PER,
                                      verbose=False):
    if mask_path_or_arr is None:
        return False

    if type(mask_path_or_arr) == str:
        if not os.path.exists(mask_path_or_arr):
            return False

        mask_arr = cv2.imread(mask_path_or_arr, cv2.IMREAD_GRAYSCALE)
    else:
        mask_arr = mask_path_or_arr

    flag = True
    stats = {}
    for c in chosen_classes:
        if np.sum(mask_arr == masks_dict[c]) < (mask_arr.shape[0] * mask_arr.shape[1] * min_pixels_mask_all_classes_per):
            stats[c] = np.sum(mask_arr == masks_dict[c])
            flag = False
            break

    if verbose:
        Image.fromarray(mask_arr * 80).show()
        print(f"Mask {mask_path_or_arr} {flag}", stats, np.unique(mask_arr))

    return flag


def make_border_of_mask(
        mask_arr,
        mask_number=None,
        pixel_mask_number=None,
        border_mask_number=None,
        border_size=None,
        combine_with_original_mask=True,
        recompute_border=False,
):
    def _make_border_of_mask(
            mask_arr,
            mask_number=seg_cf.TARGET_BORDER_CLASS['mask_number'],
            pixel_mask_number=seg_cf.MASKS[seg_cf.PIXEL_LEVEL],
            border_mask_number=seg_cf.MASKS[seg_cf.BORDER],
            border_size=seg_cf.BORDER_THICKNESS,
            combine_with_original_mask=True,
            recompute_border=False,
    ):
        if type(mask_arr) == str:
            mask_arr = cv2.imread(mask_arr, cv2.IMREAD_GRAYSCALE)
            mask_arr = cv2.resize(mask_arr, (seg_cf.HEIGHT, seg_cf.WIDTH), interpolation=cv2.INTER_NEAREST)
        else:
            mask_arr = mask_arr.copy()

        if recompute_border:
            # if we want to recompute the border, we first make the border mask number to be the mask number
            mask_arr[mask_arr != 0] = mask_number

        mul = 2
        # Pad all sides of the image with a few zeros so that border of the image is detected as well
        mask_arr = cv2.copyMakeBorder(mask_arr, border_size * mul, border_size * mul, border_size * mul, border_size * mul, cv2.BORDER_CONSTANT, value=0)

        # remove all other masks
        mask_arr[mask_arr != mask_number] = 0
        only_target_mask = mask_arr.copy()
        only_target_mask[only_target_mask != 0] = pixel_mask_number
        mask_arr[mask_arr == mask_number] = border_mask_number

        # Perform an erosion operation to shrink the object slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_size, border_size))
        eroded_mask = cv2.erode(mask_arr, kernel)

        # Subtract the eroded image from the original image
        border_mask = cv2.absdiff(mask_arr, eroded_mask)

        if combine_with_original_mask:
            border_mask = np.add(border_mask, only_target_mask)
            border_mask[border_mask == pixel_mask_number + border_mask_number] = border_mask_number

        # Remove the border pixels from all sides of the image
        border_mask = border_mask[border_size * mul:-border_size * mul, border_size * mul:-border_size * mul]

        return border_mask

    border_mask = _make_border_of_mask(
            mask_arr,
            mask_number=mask_number if mask_number is not None else seg_cf.TARGET_BORDER_CLASS['mask_number'],
            pixel_mask_number=pixel_mask_number if pixel_mask_number is not None else seg_cf.MASKS[seg_cf.PIXEL_LEVEL],
            border_mask_number=border_mask_number if border_mask_number is not None else seg_cf.MASKS[seg_cf.BORDER],
            border_size=border_size if border_size is not None else seg_cf.BORDER_THICKNESS,
            combine_with_original_mask=combine_with_original_mask if combine_with_original_mask is not None else True,
            recompute_border=recompute_border
    )

    return border_mask


# superimpose mask on an rgb image
def superimpose_mask_on_image(image, mask, mask_number=1, alpha=0.8, colour=(0, 255, 0), s_impose_mode='RGBA'):
    mask = mask.copy()
    mask = np.array(mask)

    # apply color to the binary mask
    mask[mask == mask_number] = 1
    mask[mask != mask_number] = 0
    mask_bw = mask.copy()
    mask_bw = mask_bw.astype(np.uint8)
    mask_bw = cv2.resize(mask_bw, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask = np.stack((mask, mask, mask), axis=2)
    mask = mask * np.array(colour)

    # convert to uint8
    mask = mask.astype(np.uint8)

    # apply a gaussian blur on this mask
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    print(mask.shape, image.shape)

    # superimpose the mask on the image
    image = np.array(image)
    image = image.astype(np.uint8)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    if s_impose_mode == 'colour':
        superimposed_image = cv2.addWeighted(image, alpha, mask, 1 - alpha, 0)
    elif s_impose_mode == 'RGBA':
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            image[:, :, 3] = np.ones((image.shape[0], image.shape[1]), dtype=type(image)) * 255

        mask_bw = 1 - mask_bw
        image[..., 3] *= mask_bw
        superimposed_image = image

    return superimposed_image


def super_impose_RGBA(image, mask, th=None):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    if th is not None:
        mask = (mask > th).astype(np.uint8)

    image[:, :, 3] = mask.astype(np.uint8)
    print("image shape rgba", image.shape)

    return image


# Visualize Predictions
def display(display_list, title=None):
    if title is None:
        title = ['Input Image', 'True Mask', 'Predicted Mask', 'Superimposed Mask']

    plt.figure(figsize=(10 * len(title), 10))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()


def predict_np_image(image):
    pass


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
