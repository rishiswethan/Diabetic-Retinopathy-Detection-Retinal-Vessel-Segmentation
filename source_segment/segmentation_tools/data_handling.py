import random
from abc import ABC

import albumentations as albu
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pickle

from torch.utils.data import DataLoader
import torch

import source_segment.config as cf
import source_segment.segmentation_tools.segmentation_config as seg_cf
# import source_segment.segmentation_tools_pytorch.tasm as tasm
import source_segment.segmentation_tools.utils as seg_utils
import source_segment.segmentation_tools.data_formatters.ScenenetRGBD as scenenet
import source_segment.utils as utils

# train_dataset_length = None
# valid_dataset_length = None
_dataset_path_data = {}
CLASSES_PIXEL_COUNT_DICT = {}


# initialize data
def init():
    global CLASSES_PIXEL_COUNT_DICT

    print("getting stored paths...")
    non_special_datasets = [dataset for dataset in seg_cf.CHOSEN_TRAINING_DATASETS if dataset not in seg_cf.SPECIAL_TRAINING_DATASETS]
    special_datasets = seg_cf.SPECIAL_TRAINING_DATASETS
    paths = get_stored_paths(datasets=non_special_datasets)
    x_train_dir, y_train_dir, x_valid_dir, y_valid_dir, _, __ = paths

    if seg_cf.USE_STORED_PIXEL_COUNTS:
        input("\n\n\n\n\n\nStored pixel counts will be used.        (Press Enter to continue...)")
        print("\n", "Loading stored pixel counts...")
    else:
        print("\n", "calculating pixel count for non-special datasets...")

    CLASSES_PIXEL_COUNT_DICT_NS = calculate_pixel_count_all_datasets(paths, survey_partially_per=seg_cf.SURVEY_PARTIALLY_PIXEL_COUNTS, use_stored=seg_cf.USE_STORED_PIXEL_COUNTS)
    # CLASSES_PIXEL_COUNT_DICT[cf.UNLABELED] = 0
    print("NON SPECIAL CLASSES_PIXEL_COUNT_DICT: ", CLASSES_PIXEL_COUNT_DICT_NS)

    if seg_cf.USE_SPECIAL_DATA_IN_WEIGHTS_CALC:
        print("\n", "calculating pixel count for special datasets...")
        special_paths = get_stored_paths(datasets=special_datasets)
        CLASSES_PIXEL_COUNT_DICT_S = calculate_pixel_count_all_datasets(special_paths, survey_partially_per=0.01, use_stored=True)
        # CLASSES_PIXEL_COUNT_DICT[cf.UNLABELED] = 0
        print("SPECIAL DATASET CLASSES_PIXEL_COUNT_DICT: ", CLASSES_PIXEL_COUNT_DICT_S)

        print("\n", "calculating pixel count for all datasets...")
        CLASSES_PIXEL_COUNT_DICT = seg_utils.add_2_dicts(CLASSES_PIXEL_COUNT_DICT_NS, CLASSES_PIXEL_COUNT_DICT_S)
        print("CLASSES_PIXEL_COUNT_DICT: ", CLASSES_PIXEL_COUNT_DICT, "\n")
    else:
        CLASSES_PIXEL_COUNT_DICT = CLASSES_PIXEL_COUNT_DICT_NS

    return CLASSES_PIXEL_COUNT_DICT


def crop_and_stretch(image,
                     mask,
                     max_crop_per,
                     prob_apply_aug=0.5,
                     border_mode=(True if (seg_cf.CHOSEN_MASK_TYPE == seg_cf.BORDER_MASK_TYPE) else False),
                     num_classes_border=len(seg_cf.CHOSEN_MASKS),
                     verbose=False):

    """stretch image and mask to given height and width y cropping the image and mask"""
    def crop(image, mask, _max_crop_per=max_crop_per):
        if random.random() > prob_apply_aug:
            # only apply crop if random number is greater than prob_apply_aug
            return image, mask
        else:
            if verbose:
                print("applying crop and stretch. per", end=" ")

            # randomly choose how much to crop
            crop_per = random.uniform(_max_crop_per / 8, _max_crop_per)
            if verbose:
                print(crop_per)

            # randomly choose whether to crop height or width
            crop_h, crop_w = 0, 0
            if random.random() > 0.5:
                crop_h = int(crop_per * image.shape[0])
            else:
                crop_w = int(crop_per * image.shape[1])

            # randomly choose how much to crop from each side. The amount of cropping we need will still be applied to both sides.
            left_crop = random.randint(0, crop_w)
            right_crop = crop_w - left_crop
            top_crop = random.randint(0, crop_h)
            bottom_crop = crop_h - top_crop

            image = image[
                    top_crop: image.shape[0] - bottom_crop,
                    left_crop: image.shape[1] - right_crop, :
                ]
            mask = mask[
                    top_crop: mask.shape[0] - bottom_crop,
                    left_crop: mask.shape[1] - right_crop, :
                ]

            if border_mode:
                # combine all channels of mask into one channel
                mask = np.add(mask[:, :, 1], mask[:, :, 2])

                # recompute border
                mask = seg_utils.make_border_of_mask(mask, recompute_border=True)

                # convert back to one-hot
                mask = convert_number_mask_to_onehot(mask, num_classes=num_classes_border)

            return image, mask

    def resize(image, mask, _height, _width):
        """Resize image and mask to original height and width. This will create a stretched image."""
        image = cv2.resize(image, (_height, _width), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (_height, _width), interpolation=cv2.INTER_NEAREST)
        return image, mask

    org_height, org_width = image.shape[:2]

    if verbose:
        plt.figure(figsize=(10, 40))
        print("org image.shape: ", image.shape)
        print("org mask.shape: ", mask.shape)
        plt.subplot(1, 4, 1)
        plt.imshow(image)
        plt.subplot(1, 4, 2)
        plt.imshow(mask)
        # plt.show()

    # crop
    image, mask = crop(image, mask)

    # resize
    image, mask = resize(image, mask, org_height, org_width)

    if verbose:
        print("image.shape: ", image.shape)
        print("mask.shape: ", mask.shape)
        plt.subplot(1, 4, 3)
        plt.imshow(image)
        plt.subplot(1, 4, 4)
        plt.imshow(mask)
        plt.show()

    return image, mask


# define heavy augmentations
def get_training_augmentation(height, width):
    def _get_training_augmentation(height, width):
        train_transform = [

            albu.HorizontalFlip(p=0.5),

            albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=0, shift_limit=0.0, p=0.75, border_mode=0),
            # albu.RandomResizedCrop(height=height, width=width, scale=(0.5, 1.5), ratio=(0.75, 1.3333333333333333), interpolation=1, always_apply=True, p=1),

            albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
            albu.RandomCrop(height=height, width=width, always_apply=True, p=0.7),

            albu.IAAAdditiveGaussianNoise(p=0.3),
            albu.IAAPerspective(p=0.5),
            albu.Rotate(limit=60, p=1.0),

            albu.OneOf(
                [
                    albu.CLAHE(p=1),
                    albu.RandomBrightness(p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.IAASharpen(p=1),
                    albu.Blur(blur_limit=3, p=1),
                    albu.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.RandomContrast(p=1),
                    albu.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        ]
        return albu.Compose(train_transform)

    # def _custom_augmentations(image, mask):
    #     augs = _get_training_augmentation(height, width)(image=image, mask=mask)
    #     image, mask = augs["image"], augs["mask"]
    #
    #     image, mask = crop_and_stretch(image, mask, max_crop_per=0.1, prob_apply_aug=0.5, verbose=False)
    #
    #     return {"image": image, "mask": mask}

    return _get_training_augmentation(height, width)


def convert_number_mask_to_onehot(mask, num_classes=None, masks=seg_cf.MASKS, border_mode=seg_cf.CHOSEN_MASK_TYPE == seg_cf.BORDER_MASK_TYPE):
    """Convert number mask to one-hot mask"""
    if num_classes is None:
        _num_classes = len(np.unique(mask))
    else:
        _num_classes = num_classes

    mask = mask.astype(np.uint8)
    mask_ = mask.copy()
    # print("mask.shape: ", mask.shape, "mask unique: ", np.unique(mask))
    mask = np.eye(_num_classes)[mask]

    # if border_mode:
    #     # make sure pixel values overlap with border mask
    #     mask[..., masks[seg_cf.PIXEL_LEVEL]] = np.where(mask_ > 0, 1, 0)

    # if mask has less than num_class channels, add the missing channels
    if num_classes is not None:
        if mask.shape[2] < num_classes:
            mask = np.concatenate((mask, np.zeros(mask.shape[:2] + (_num_classes - mask.shape[2],))), axis=2)

    # if border_mode:
    #     mask_small_border = seg_utils.make_border_of_mask(mask_, border_size=seg_cf.BORDER_THICKNESS // 2, recompute_border=True)
    #     mask[..., 2] = np.where(mask_small_border == 2, 1, 0)

    return mask


def get_validation_augmentation(height, width):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # import cv2.cv2 as cv2
        # A.PadIfNeeded(height, width, border_mode=cv2.BORDER_ISOLATED),
        albu.Resize(height, width, always_apply=True, interpolation=cv2.INTER_NEAREST)
    ]
    return albu.Compose(test_transform)


def data_get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
    ]
    return albu.Compose(_transform)


################################################################################
# Data Generator
################################################################################


def _get_file_paths(train_or_test, one_per_folder_datasets=seg_cf.SPECIAL_TRAINING_DATASETS):
    # global per_dataset_images_fps, of_image_names, of_mask_names, train_dataset_length, valid_dataset_length
    # get number of unique folders in the special datasets
    per_dataset_images_fps = {}
    for dataset in one_per_folder_datasets:
        if dataset in seg_cf.CHOSEN_TRAINING_DATASETS:
            # get the unique folders in the special datasets
            print("Getting unique folders for", dataset, "dataset")
            paths = scenenet.get_unique_folders(cf.TRAINING_FOLDER_PATHS[dataset])
            train_image_folders, train_mask_folders, test_image_folders, test_mask_folders, of_image_names, of_mask_names = paths
            assert len(of_image_names) == len(of_mask_names), "The number of images and masks in the one folder dataset must be the same " + str(len(of_image_names)) + " " + str(len(of_mask_names))
            assert len(train_image_folders) == len(train_mask_folders), "The number of images and masks in the train folder dataset must be the same " + str(len(train_image_folders)) + " " + str(len(train_mask_folders))
            assert len(test_image_folders) == len(test_mask_folders), "The number of images and masks in the test folder dataset must be the same " + str(len(test_image_folders)) + " " + str(len(test_mask_folders))
            print("done")

            # change the folder list of the special datasets to the unique folders
            if train_or_test == "train":
                per_dataset_images_fps[dataset] = [train_image_folders, train_mask_folders, of_image_names, of_mask_names]
            elif train_or_test == "test":
                per_dataset_images_fps[dataset] = [test_image_folders, test_mask_folders, of_image_names, of_mask_names]

    # get the rest of the datasets
    for dataset in seg_cf.CHOSEN_TRAINING_DATASETS:
        if dataset not in one_per_folder_datasets:
            # get the paths of the dataset that was loaded in load_dataset_paths()
            train_image_paths_, train_mask_paths_, test_image_paths_, test_mask_paths_ = _dataset_path_data[dataset]

            # load the other datasets normally
            if train_or_test == "train":
                per_dataset_images_fps[dataset] = [train_image_paths_, train_mask_paths_]
            elif train_or_test == "test":
                per_dataset_images_fps[dataset] = [test_image_paths_, test_mask_paths_]

    # find the length of each dataset
    each_dataset_len = [len(per_dataset_images_fps[dataset][0]) for dataset in per_dataset_images_fps]
    total_len = sum(each_dataset_len)
    print("datasets", per_dataset_images_fps.keys())
    print("each_dataset_len", each_dataset_len)

    # This is where global lengths are set. We currently use it to calculate steps per epoch in train and test
    # if train_or_test == "train":
    #     train_dataset_length = total_len
    # elif train_or_test == "test":
    #     valid_dataset_length = total_len

    return per_dataset_images_fps, total_len


def preprocess_image_mask(image, mask=None):
    image = image.astype(np.float32)
    image /= 255.0
    sample = {
        "image": image,
        "mask": mask
    }

    return sample


def process_image_label(images_paths,
                        masks_paths,
                        classes,
                        binary_mode=seg_cf.BINARY_MODE,
                        all_classes=seg_cf.MASKS,
                        augmentation=None,
                        preprocessing=preprocess_image_mask,
                        special_dataset=False,
                        border_mode_flag=seg_cf.CHOSEN_MASK_TYPE,
                        verbose=0):
    # resize the mask to the size of the image
    mask = cv2.imread(masks_paths, 0)
    mask = cv2.resize(mask, (seg_cf.HEIGHT, seg_cf.WIDTH), interpolation=cv2.INTER_NEAREST)
    if special_dataset and border_mode_flag == seg_cf.BORDER_MASK_TYPE:
        # special dataset doesn't have the border detected. We should do it here
        mask = seg_utils.make_border_of_mask(masks_paths)

    class_values = [all_classes[cls] for cls in classes]
    # Chosen class values won't have zero(background) value if mode is not binary classification, so we add it
    if 0 not in class_values:
        class_values = [0] + class_values
    class_values = sorted(class_values)
    missing_class_values = [v for v in all_classes.values() if v not in class_values]

    # read data
    image = cv2.imread(images_paths)
    image = cv2.resize(image, (seg_cf.HEIGHT, seg_cf.WIDTH), interpolation=cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for class_val in missing_class_values:
        mask[mask == class_val] = 0

    # remove all values that are not in the chosen classes from the mask
    unknown_class_values = [v for v in np.unique(mask) if v not in class_values]
    for class_val in unknown_class_values:
        mask[mask == class_val] = 0

    # one-hot encoding.
    if border_mode_flag:
        mask = convert_number_mask_to_onehot(mask, num_classes=len(class_values))
    else:
        masks = np.zeros((mask.shape[0], mask.shape[1], len(class_values)), dtype=np.float32)
        for i, key in enumerate(all_classes):
            if key in classes:
                class_val = class_values[i]
                if verbose:
                    print("class_val", class_val, key, class_values.index(class_val))
                masks[..., i] = (mask == int(class_val))
        mask = masks

    # apply augmentations
    if augmentation:
        sample = augmentation(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']

    # apply preprocessing
    if preprocessing:
        sample = preprocessing(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']

    if verbose:
        print("image", image.shape, "mask", mask.shape)
        # show all layers in a loop
        plt.figure(figsize=(15, 15))
        plt.subplot(1, mask.shape[-1] + 1, 1)
        plt.title("image")
        plt.imshow((image * 255).astype(np.uint8))
        for i in range(mask.shape[-1]):
            disp_mask = mask[..., i].copy() * 255
            disp_mask = disp_mask.astype(np.uint8)
            plt.subplot(1, mask.shape[-1] + 1, i + 2)
            plt.title("layer " + str(i))
            plt.imshow(disp_mask)
        plt.show()
        plt.close()

    # for pytorch we need to transpose the image and mask to (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    mask = np.transpose(mask, (2, 0, 1))

    return image, mask


class CustomImagePathGenerator:
    def __init__(self,
                 train_or_test,
                 per_dataset_images_fps,
                 shuffle=True,
                 disallow_special_dataset_in_test=seg_cf.DISALLOW_SPECIAL_DATASETS_IN_TEST,
                 one_per_folder_datasets=seg_cf.SPECIAL_TRAINING_DATASETS,
                 test_datasets=seg_cf.TEST_SET_DATASETS,
                 verbose=False):

        self.train_or_test = train_or_test
        self.per_dataset_images_fps = per_dataset_images_fps.copy()
        self.shuffle = shuffle
        self.one_per_folder_datasets = one_per_folder_datasets.copy()
        self.verbose = verbose

        # remove special dataset from the list of datasets if test
        if (self.train_or_test == "test") and disallow_special_dataset_in_test:
            for dataset in self.one_per_folder_datasets:
                self.per_dataset_images_fps.pop(dataset)

        # only use these datasets for test if test_datasets is not empty
        if (self.train_or_test == "test") and (len(test_datasets) > 0):
            for dataset in self.per_dataset_images_fps.copy():
                if dataset not in test_datasets:
                    self.per_dataset_images_fps.pop(dataset)

        # find the length of each dataset
        each_dataset_len = [len(self.per_dataset_images_fps[dataset][0]) for dataset in self.per_dataset_images_fps]
        self.total_len = sum(each_dataset_len)
        print("datasets", self.per_dataset_images_fps.keys())
        print("each_dataset_len", each_dataset_len)

        # get shuffled indices of the total length
        indices = list(range(self.total_len))
        if shuffle:
            indices = seg_utils.shuffle_train_data(indices, seed=random.randint(0, 10000))
        self.indices = indices
        self.iter = 0

        # get the range of each dataset so that we can choose the dataset based on the index
        dataset_ranges = {}
        range_end = 0
        for i, dataset in enumerate(self.per_dataset_images_fps):
            dataset_ranges[dataset] = [range_end, range_end + each_dataset_len[i]]
            range_end += each_dataset_len[i]
        self.dataset_ranges = dataset_ranges
        print("dataset_ranges", dataset_ranges)

    def __len__(self):
        return self.total_batches

    def next(self, special_dataset=False):
        if self.iter >= self.total_len:
            # We are skipping some bad data, so a full epoch may iterate over the same data multiple times. We thus need to shuffle the data again, just to be safe.
            self.on_epoch_end()
            # reset the iterator
            self.iter = 0

        index = self.indices[self.iter]
        # special_dataset not false indicated that a file from the special dataset has been rejected. We need to find the next one.
        if special_dataset:
            data_to_send, dataset_name = special_dataset
            image_path, mask_path = data_to_send

            if self.verbose:
                print("Mask rejected, finding another one from the same folder")
                print(image_path)

            # List of image names and mask names only, without the folder path
            of_image_names = self.per_dataset_images_fps[dataset_name][2]
            of_mask_names = self.per_dataset_images_fps[dataset_name][3]

            # For the special datasets, we need to choose a random image from the folder. The values of per_dataset_images_fps are the folders.
            try:
                random_file_name = random.randint(0, len(of_image_names) - 1)  # -1 because randint is inclusive
            except:
                print("Error in random.randint", len(of_image_names), dataset_name, of_image_names)

            # Choose a random image, and it's corresponding mask from the folder
            random_image = of_image_names[random_file_name]
            random_mask = of_mask_names[random_file_name]

            # find another file from the same folder
            image_path = image_path.replace(image_path.split(os.sep)[-1], random_image)
            mask_path = mask_path.replace(mask_path.split(os.sep)[-1], random_mask)

            if self.verbose:
                print("New image path")
                print(image_path, "\n")

            data_to_send = [image_path, mask_path]

            return data_to_send, dataset_name

        # choose the dataset based on the index
        for dataset in self.dataset_ranges:
            if (index >= self.dataset_ranges[dataset][0]) and (index <= self.dataset_ranges[dataset][1] - 1):
                chosen_dataset = dataset
                break

        # index of the second dataset, for example, will start at 0. But, index being more 0'th dataset length is why we moved to the next one. We are correcting for this here.
        correct_index = index - self.dataset_ranges[chosen_dataset][0]

        try:
            if chosen_dataset in self.one_per_folder_datasets:
                # List of image names and mask names only, without the folder path
                of_image_names = self.per_dataset_images_fps[chosen_dataset][2]
                of_mask_names = self.per_dataset_images_fps[chosen_dataset][3]

                try:
                    random_file_name = random.randint(0, len(of_image_names) - 1)  # -1 because randint is inclusive
                except:
                    print("Error in random.randint", len(of_image_names), chosen_dataset, of_image_names)
                    raise ValueError

                # Choose a random image, and it's corresponding mask from the folder
                random_image = of_image_names[random_file_name]
                random_mask = of_mask_names[random_file_name]

                # [image name and mask name]
                data_to_send = [self.per_dataset_images_fps[chosen_dataset][0][correct_index] + random_image,
                                self.per_dataset_images_fps[chosen_dataset][1][correct_index] + random_mask]
            else:
                data_to_send = [self.per_dataset_images_fps[chosen_dataset][0][correct_index],
                                self.per_dataset_images_fps[chosen_dataset][1][correct_index]]
        except IndexError:
            print("chosen_dataset", chosen_dataset)
            print("correct_index", correct_index)
            print("index", index)
            print("per_dataset_images_fps", len(self.per_dataset_images_fps[chosen_dataset]))
            print(f"per_dataset_images_fps[{0}]", len(self.per_dataset_images_fps[chosen_dataset][0]))
            print(f"per_dataset_images_fps[{1}]", len(self.per_dataset_images_fps[chosen_dataset][1]))
            raise IndexError

        self.iter += 1
        return data_to_send, chosen_dataset

    def __next__(self):
        special_dataset = False
        special_attempt_cnt = 0
        while True:
            data_to_send, chosen_dataset = self.next(special_dataset=special_dataset)
            image_fp, mask_fp = data_to_send

            if (seg_cf.CHOSEN_MASK_TYPE == seg_cf.BORDER_MASK_TYPE) and (chosen_dataset in self.one_per_folder_datasets):
                classes_to_check_min_pixels = [seg_cf.TARGET_BORDER_CLASS["name"]]
                masks_dict_for_min_pixels = {}
                for classes in seg_cf.ALL_ORIGINAL_MASKS:
                    masks_dict_for_min_pixels[classes["name"]] = classes["mask_number"]
                min_pixels_percentage = seg_cf.SPECIAL_DATASET_MIN_PIXELS
            else:
                classes_to_check_min_pixels = seg_cf.CHECK_MIN_PIXELS_CLASSES
                masks_dict_for_min_pixels = seg_cf.MASKS
                min_pixels_percentage = seg_cf.MIN_PIXELS_MASK_ALL_CLASSES_PER

            # check if the data_to_send has atleast a minimum % of pixel of all classes
            if seg_utils.check_presence_of_min_pixels_mask(mask_fp,
                                                           chosen_classes=classes_to_check_min_pixels,
                                                           masks_dict=masks_dict_for_min_pixels,
                                                           min_pixels_mask_all_classes_per=min_pixels_percentage,
                                                           verbose=False):
                return data_to_send, chosen_dataset
            else:
                if chosen_dataset in self.one_per_folder_datasets:
                    if special_attempt_cnt < 20:
                        special_dataset = data_to_send, chosen_dataset
                        special_attempt_cnt += 1
                    else:
                        special_dataset = False
                        special_attempt_cnt = 0
                continue

    def on_epoch_end(self):
        # shuffle the indices after each epoch
        if self.shuffle:
            self.indices = seg_utils.shuffle_train_data(self.indices, seed=random.randint(0, 10000))


class DataGenerator(torch.utils.data.Dataset):
    def __init__(
            self,
            train_or_test,
            batch_size,
            height, width,
            classes,
            augmentation,
            wwo_aug=False,
            prob_apply_aug=1.0,
            shuffle=True,
            one_per_folder_datasets=seg_cf.SPECIAL_TRAINING_DATASETS,
            seed=None,
            verbose=False
    ):
        per_dataset_images_fps, total_len = _get_file_paths(train_or_test)
        self.image_label_path_generator = CustomImagePathGenerator(
            train_or_test,
            per_dataset_images_fps=per_dataset_images_fps,
            shuffle=shuffle,
            verbose=verbose
        )
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.classes = classes
        self.augmentation = augmentation
        self.wwo_aug = wwo_aug
        self.one_per_folder_datasets = one_per_folder_datasets
        self.prob_apply_aug = prob_apply_aug
        self.seed = seed
        self.verbose = verbose
        self.train_or_test = train_or_test

        self.per_dataset_images_fps = per_dataset_images_fps
        self.train_or_val_dataset_length = total_len

    def __len__(self):
        return self.train_or_val_dataset_length

    def __getitem__(self, idx):
        (image_path, label_path), dataset_referred = next(self.image_label_path_generator)
        special_dataset_flag = (dataset_referred in self.one_per_folder_datasets)

        if random.random() <= self.prob_apply_aug:
            image, label = process_image_label(image_path, label_path, classes=self.classes, augmentation=self.augmentation, special_dataset=special_dataset_flag, verbose=self.verbose)
        else:
            image, label = process_image_label(image_path, label_path, classes=self.classes, special_dataset=special_dataset_flag, verbose=self.verbose)

        if self.verbose:
            print("Image: ", image_path)
            print("Label: ", label_path)

        # return tf.convert_to_tensor(images), tf.convert_to_tensor(labels, tf.float32)
        # print("images.shape: ", image.shape, type(image))
        return image, label


class SimpleGenerator:
    def __init__(self, folder_path, verbose=False):
        self.folder_path = folder_path
        self.image_path_generator = SimpleImagePathGenerator(folder_path, verbose=verbose)

    def __len__(self):
        return len(os.listdir(self.folder_path))

    def __getitem__(self, item):
        while True:
            image_path, flag = next(self.image_path_generator)
            if flag:
                image_array = cv2.imread(image_path)
                image_array = image_array[..., :3]
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                # image_array = cv2.resize(image_array, (seg_cf.HEIGHT, seg_cf.WIDTH), interpolation=cv2.INTER_NEAREST)
                image_array = preprocess_image_mask(image_array)["image"]

                return image_array, image_path
            else:
                return None

class SimpleImagePathGenerator:
    def __init__(self, folder_path, verbose=False):
        self.folder_path = folder_path
        self.verbose = verbose
        self.image_fps = []
        self.last_index = 0
        self._get_file_paths()

    def _get_file_paths(self):
        print("file list:-")
        for file_name in os.listdir(self.folder_path):
            if os.path.isfile(os.path.join(self.folder_path, file_name)):
                print("File: ", file_name)
                self.image_fps.append(os.path.join(self.folder_path, file_name))

    def __len__(self):
        return len(self.image_fps)

    def __next__(self):
        return_path = self.image_fps[self.last_index]
        self.last_index += 1
        print("return_path: ", return_path)
        if self.last_index > len(self.image_fps):
            return "end", False
        else:
            return return_path, True


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def _show_sample_predictions_border(
        sample_image,
        sample_mask,
        model,
        display_image=True,
        class_weights=None,
        detection_threshold=seg_cf.PREDICTION_TH,
):
    output_model = model.predict((sample_image[tf.newaxis, ...]))
    print("output_model shape", output_model.shape, np.sort(np.unique(output_model)))

    scce = tf.keras.losses.CategoricalCrossentropy()
    print("SparseCategoricalCrossentroy: " + str(scce(sample_mask, output_model[0]).numpy()))
    print("Iou-Score: " + str(tasm.iou_score(sample_mask, output_model[0]).numpy()))
    print("categorical Focal Dice Loss: " + str(tasm.categorical_focal_dice_loss(sample_mask, output_model[0]).numpy()))
    if display_image:
        sample_mask_disp = output_model[0]
        # sample_mask_disp = np.array(sample_mask)

        # extract the mask of interest and superimpose it on the image
        # sample_mask = np.argmax(sample_mask, axis=-1)
        # sample_mask = np.expand_dims(sample_mask, axis=-1)
        sample_mask_1 = sample_mask[..., 1]
        sample_mask_1 = np.expand_dims(sample_mask_1, axis=-1)
        # sample_mask_2 = sample_mask[..., 2]
        # sample_mask_2 = np.expand_dims(sample_mask_2, axis=-1)

        display_list = [sample_image, sample_mask_1, create_mask(output_model)]

        print("sample_mask shape", sample_mask_disp.shape, np.sort(np.unique(sample_mask_disp)))

        # argmax returns the INDEX of the max value in the array channels
        arg_max_arrays = np.argmax(sample_mask_disp[..., 0:3], axis=-1)

        super_imposed = sample_image * 255.
        for i in range(1, sample_mask_disp.shape[-1]):
            layer_of_interest = sample_mask_disp[..., i].copy()
            # layer_of_interest[layer_of_interest > 0.5] = 1

            disabled_pixels = layer_of_interest <= detection_threshold
            layer_of_interest[arg_max_arrays != i] = 0
            layer_of_interest[arg_max_arrays == i] = 1

            layer_of_interest[disabled_pixels] = 0

            # Image.fromarray(layer_of_interest.astype(np.uint8) * 255).show()

            print("layer_of_interest shape", layer_of_interest.shape, np.sort(np.unique(layer_of_interest)))
            # colours -> red, green, blue, yellow, magenta, cyan, white
            colours = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)
            colour = colours[i]

            super_imposed = seg_utils.superimpose_mask_on_image(super_imposed, layer_of_interest, colour=colour).copy()

        # super_imposed[..., -1] = super_imposed[..., -1] * 255.
        display_list.append(super_imposed)
        seg_utils.display(display_list, title=['Input Image', 'True Mask1', 'Predicted Mask', 'Superimposed Mask'])


def _show_sample_predictions_pixelwise(
        sample_image,
        sample_mask,
        model,
        display_image=True,
        class_weights=None,
):
    output_model = model(sample_image[tf.newaxis, ...])
    print(output_model.numpy().shape)

    output_mask = create_mask(output_model)
    # print(sample_mask.shape)

    scce = tf.keras.losses.CategoricalCrossentropy()
    print("SparseCategoricalCrossentroy: " + str(scce(sample_mask, output_model[0]).numpy()))
    print("Iou-Score: " + str(tasm.iou_score(sample_mask, output_model[0]).numpy()))
    print("categorical Focal Dice Loss: " + str(tasm.categorical_focal_dice_loss(sample_mask, output_model[0]).numpy()))
    if display_image:
        seg_utils.display([sample_image, sample_mask, K.one_hot(K.squeeze(output_mask, axis=-1), 3)])


def show_sample_predictions(
        sample_image,
        sample_mask,
        model,
        display_image=True,
        class_weights=None,
):
    if seg_cf.CHOSEN_MASK_TYPE == seg_cf.BORDER_MASK_TYPE or seg_cf.BINARY_MODE:
        _show_sample_predictions_border(sample_image, sample_mask, model, display_image, class_weights)
    elif seg_cf.CHOSEN_MASK_TYPE == seg_cf.PIXEL_LEVEL_MASK_TYPE:
        _show_sample_predictions_pixelwise(sample_image, sample_mask, model, display_image, class_weights)  # using the same function as border for now
    else:
        raise ValueError("CHOSEN_MASK_TYPE not supported")


def calculate_pixel_count_all_datasets(paths, use_stored=False, survey_partially_per=1.0):
    if os.path.exists(cf.PIXEL_COUNT_PATH) and use_stored:
        f = open(cf.PIXEL_COUNT_PATH, 'rb')
        pixel_count = pickle.load(f)
        f.close()

        print("pixel count", pixel_count)
    else:
        train_image_paths, train_mask_paths, test_image_paths, test_mask_paths, _, __ = paths
        pixel_count = seg_utils.get_dataset_pixel_counts(masks_path=train_mask_paths, survey_partially_per=survey_partially_per)
        print("new pixel count", pixel_count)

        # save pixel count
        f = open(cf.PIXEL_COUNT_PATH, 'wb')
        pickle.dump(pixel_count, f)
        f.close()

    return pixel_count


# load paths pickles for all datasets
def get_stored_paths(datasets=seg_cf.CHOSEN_TRAINING_DATASETS):
    def concatenate_lists(lists):
        return [item for sublist in lists for item in sublist]

    train_image_paths, train_mask_paths, test_image_paths, test_mask_paths = [], [], [], []
    per_dataset_image_paths, per_dataset_mask_paths = {}, {}
    for dataset in datasets:
        per_dataset_image_paths[dataset], per_dataset_mask_paths[dataset] = [], []

        print(dataset, "loading")
        f = open(cf.TRAINING_FOLDER_PATHS[dataset] + "paths_pkl" + os.sep + "paths.pkl", 'rb')
        _dataset_path_data[dataset] = pickle.load(f)
        f.close()
        train_image_paths_, train_mask_paths_, test_image_paths_, test_mask_paths_ = _dataset_path_data[dataset]
        print("loaded")

        per_dataset_image_paths[dataset] = train_image_paths_
        per_dataset_mask_paths[dataset] = train_mask_paths_

        print("concatenating")
        train_image_paths = concatenate_lists([train_image_paths, train_image_paths_])
        train_mask_paths = concatenate_lists([train_mask_paths, train_mask_paths_])
        test_image_paths = concatenate_lists([test_image_paths, test_image_paths_])
        test_mask_paths = concatenate_lists([test_mask_paths, test_mask_paths_])
        print("concatenated")

        print("shuffling")
        train_image_paths = seg_utils.shuffle(train_image_paths)
        train_mask_paths = seg_utils.shuffle(train_mask_paths)
        test_image_paths = seg_utils.shuffle(test_image_paths)
        test_mask_paths = seg_utils.shuffle(test_mask_paths)
        print("shuffled")

        print("train_image_paths: ", len(train_image_paths_))
        print("train_mask_paths: ", len(train_mask_paths_))
        print("test_image_paths: ", len(test_image_paths_))
        print("test_mask_paths: ", len(test_mask_paths_))

        f.close()

    print("\nAll datasets")
    print("train_image_paths: ", len(train_image_paths))
    print("train_mask_paths: ", len(train_mask_paths))
    print("test_image_paths: ", len(test_image_paths))
    print("test_mask_paths: ", len(test_mask_paths))
    for dataset in per_dataset_image_paths:
        print(dataset, "train_image_paths: ", len(per_dataset_image_paths[dataset]))
        print(dataset, "train_mask_paths: ", len(per_dataset_mask_paths[dataset]))

    return train_image_paths, train_mask_paths, test_image_paths, test_mask_paths, per_dataset_image_paths, per_dataset_mask_paths


if __name__ == '__main__':
    # pixel_cnts = calculate_pixel_count_all_datasets()
    #
    # MODEL_CLASSES = seg_cf.CHOSEN_MASKS.copy()
    # MODEL_CLASSES.remove(seg_cf.UNLABELED)
    # print(seg_utils.get_balancing_class_weights(MODEL_CLASSES, pixel_cnts))

    # test the image data generator
    get_stored_paths()

    per_dataset_images_fps, total_len = _get_file_paths('test')
    generator = CustomImagePathGenerator(
        train_or_test='train',
        shuffle=True,
        per_dataset_images_fps=per_dataset_images_fps,
        verbose=True,
    )
    for i in range(10):
        data, dataset = next(generator)
        print(dataset)
        print(data)
