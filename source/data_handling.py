import random
import albumentations as albu
import numpy as np
import os
import cv2
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

import source.config as cf
import source.utils as utils

DATASETS_IN_USE = cf.DATASETS_IN_USE
TEST_DATASETS = cf.TEST_DATASETS
DATA_FOLDERS = cf.DATA_FOLDERS

SQUARE_SIZE = cf.SQUARE_SIZE


def get_training_augmentation(height, width):
    def _get_training_augmentation(height, width):
        train_transform = [

            # albu.HorizontalFlip(p=0.5),

            # albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=0, shift_limit=0.0, p=0.75, border_mode=0),
            # albu.RandomResizedCrop(height=height, width=width, scale=(0.5, 1.5), ratio=(0.75, 1.3333333333333333), interpolation=1, always_apply=True, p=1),

            albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
            # albu.RandomCrop(height=height, width=width, always_apply=True, p=0.7),

            albu.IAAAdditiveGaussianNoise(p=0.3),
            # albu.IAAPerspective(p=0.5),
            # albu.Rotate(limit=60, p=1.0),

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

    return _get_training_augmentation(height, width)


def get_validation_augmentation(height, width):
    def _get_validation_augmentation(height, width):
        """Add paddings to make image shape divisible by 32"""
        test_transform = [
            albu.PadIfNeeded(height=height, width=width, always_apply=True, border_mode=0),
        ]
        return albu.Compose(test_transform)

    return _get_validation_augmentation(height, width)


def image_paths(dataset, train_or_test, data_folders=DATA_FOLDERS):
    paths = []
    for image in os.listdir(os.path.join(data_folders['training_data'], dataset, train_or_test)):
        paths.append(os.path.join(data_folders['training_data'], dataset, train_or_test, image))

    return paths


def process_image(image_path, square_size, augmentation):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # pad image to square
    height, width, _ = image.shape
    if height > width:
        pad = (height - width) // 2
        image = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_CONSTANT)
    elif width > height:
        pad = (width - height) // 2
        image = cv2.copyMakeBorder(image, pad, pad, 0, 0, cv2.BORDER_CONSTANT)

    image = cv2.resize(image, (square_size, square_size))

    if augmentation is not None:
        image = augmentation(image=image)['image']

    image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W) for pytorch
    image = image / 255.0

    label = utils.get_label_from_path(image_path)

    return image, label


def get_class_wise_cnt(paths):
    per_class_cnt = {}
    for path in paths:
        label = utils.get_label_from_path(path, return_one_hot=False)
        if label in per_class_cnt:
            per_class_cnt[label] += 1
        else:
            per_class_cnt[label] = 1

    return per_class_cnt


class CustomImagePathGenerator:
    def __init__(
            self,
            train_or_test,
            train_datasets=DATASETS_IN_USE,
            test_datasets=TEST_DATASETS,
            shuffle=True,
            verbose=False
    ):
        self.train_or_test = train_or_test
        self.shuffle = shuffle
        self.verbose = verbose
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.def_image_paths = []
        self.last_iter_index = 0
        self.image_paths_length = 0
        self.per_class_cnt = {}

        assert self.train_or_test in ['train', 'test'], "train_or_test must be either 'train' or 'test'"

        self.initialise_paths()

    def initialise_paths(self):
        self.def_image_paths = []
        for dataset in self.test_datasets:
            paths = image_paths(dataset, self.train_or_test)
            self.def_image_paths.extend(paths)

        if self.shuffle:
            random.shuffle(self.def_image_paths)

        self.image_paths_length = len(self.def_image_paths)
        self.per_class_cnt = get_class_wise_cnt(self.def_image_paths)

    def __len__(self):
        return self.image_paths_length

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.def_image_paths)

        for image_path in self.def_image_paths:
            yield image_path

    def __next__(self):
        if self.last_iter_index == len(self.def_image_paths):
            random.shuffle(self.def_image_paths)
            self.last_iter_index = 0

        image_path = self.def_image_paths[self.last_iter_index]
        self.last_iter_index += 1

        return image_path


class DataGenerator(torch.utils.data.Dataset):
    def __init__(
            self,
            train_or_test,
            augmentation=None,
            image_square_size=SQUARE_SIZE,
            shuffle=True,
            verbose=False
    ):
        self.train_or_test = train_or_test
        self.image_square_size = image_square_size
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.verbose = verbose

        assert self.train_or_test in ['train', 'test'], "train_or_test must be either 'train' or 'test'"

        self.image_label_path_generator = CustomImagePathGenerator(train_or_test=self.train_or_test, shuffle=self.shuffle, verbose=self.verbose)
        self.per_class_cnt = self.image_label_path_generator.per_class_cnt

    def __len__(self):
        return len(self.image_label_path_generator)

    def __getitem__(self, idx):
        image_path = next(self.image_label_path_generator)

        image, label = process_image(image_path, square_size=self.image_square_size, augmentation=self.augmentation)

        if self.verbose:
            print("Image: ", image_path)
            print("Label: ", label)

        # return image, label
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# datagen = DataGenerator(train_or_test='train', augmentation=get_training_augmentation(SQUARE_SIZE, SQUARE_SIZE), verbose=True)
# for image, label in datagen:
#     print(image.shape)
#     print(label)
#
#     plt.imshow(np.transpose(image, (1, 2, 0)))
#     plt.show()
