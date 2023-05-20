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


def get_training_augmentation(height, width, use_geometric_aug=True, use_colour_aug=False, prob_each_aug=0.6):
    def _get_training_augmentation(height, width):
        """
        RandomBrightnessContrast: Adjusts the brightness and contrast of the image randomly.

        RGBShift: Changes the values of the RGB channels.

        ChannelShuffle: Changes the order of the image channels.

        CoarseDropout: Sets rectangular regions within the image to zero.

        Cutout: Sets rectangular regions within the image to the mean pixel value.

        GaussNoise: Adds Gaussian noise to the image.

        ImageCompression: Decreases the quality of the image.

        InvertImg: Inverts the colors of the image.

        MedianBlur: Blurs the image using a median filter.

        ToGray: Converts the image to grayscale.
        """


        train_transform = [
            albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),

            albu.IAAAdditiveGaussianNoise(p=prob_each_aug),

            albu.CLAHE(p=prob_each_aug),

            albu.RandomBrightness(p=prob_each_aug),

            albu.RandomGamma(p=prob_each_aug),

            albu.IAASharpen(p=prob_each_aug),

            albu.Blur(blur_limit=3, p=prob_each_aug) if use_colour_aug else None,

            albu.MotionBlur(blur_limit=3, p=prob_each_aug) if use_colour_aug else None,

            albu.RandomContrast(p=prob_each_aug),

            albu.HueSaturationValue(p=prob_each_aug) if use_colour_aug else None,

            albu.RandomBrightnessContrast(p=prob_each_aug),

            albu.GaussNoise(p=prob_each_aug),

            albu.MedianBlur(blur_limit=5, p=prob_each_aug) if use_colour_aug else None,

            albu.InvertImg(p=prob_each_aug) if use_colour_aug else None,

            albu.ToGray(p=prob_each_aug) if use_colour_aug else None,

            # geometric transforms
            albu.HorizontalFlip(p=prob_each_aug) if use_geometric_aug else None,

            albu.VerticalFlip(p=prob_each_aug) if use_geometric_aug else None,

            albu.Rotate(limit=180, p=prob_each_aug) if use_geometric_aug else None,

            albu.IAAPerspective(p=prob_each_aug) if use_geometric_aug else None,

            albu.RandomCrop(height=height, width=width, always_apply=True, p=prob_each_aug) if use_geometric_aug else None,

            albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=0, shift_limit=0.0, p=prob_each_aug, border_mode=0) if use_geometric_aug else None,

            albu.RandomResizedCrop(height=height, width=width, scale=(0.5, 1.5), ratio=(0.75, 1.3333333333333333), interpolation=1, always_apply=True, p=prob_each_aug) if use_geometric_aug else None,

            albu.RGBShift(p=prob_each_aug) if use_geometric_aug else None,

            albu.ChannelShuffle(p=prob_each_aug) if use_geometric_aug else None,

            albu.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=prob_each_aug) if use_geometric_aug else None,

            albu.Cutout(num_holes=8, max_h_size=8, max_w_size=8, p=prob_each_aug) if use_geometric_aug else None,
        ]

        # remove None from list
        train_transform = [i for i in train_transform if i is not None]

        return albu.Compose(train_transform)

    return _get_training_augmentation(height, width)


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
            verbose=False,
            prob_apply_augmentation=1.0
    ):
        self.train_or_test = train_or_test
        self.image_square_size = image_square_size
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.verbose = verbose
        self.prob_apply_augmentation = prob_apply_augmentation

        assert self.train_or_test in ['train', 'test'], "train_or_test must be either 'train' or 'test'"

        self.image_label_path_generator = CustomImagePathGenerator(train_or_test=self.train_or_test, shuffle=self.shuffle, verbose=self.verbose)
        self.per_class_cnt = self.image_label_path_generator.per_class_cnt

    def __len__(self):
        return len(self.image_label_path_generator)

    def __getitem__(self, idx):
        # Compute the actual index
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            actual_idx = idx
        else:
            # in a worker process
            # This assumes that each worker gets an equal share of the data.
            per_worker = len(self) // worker_info.num_workers
            residual = len(self) % worker_info.num_workers
            if worker_info.id < residual:
                start_idx = (per_worker + 1) * worker_info.id
            else:
                start_idx = per_worker * worker_info.id + residual
            actual_idx = start_idx + idx

        if actual_idx >= len(self.image_label_path_generator):
            actual_idx = random.randint(0, len(self.image_label_path_generator) - 1)

        image_path = self.image_label_path_generator.def_image_paths[actual_idx]

        if self.prob_apply_augmentation < random.random():
            image, label = process_image(image_path, square_size=self.image_square_size, augmentation=self.augmentation)
        else:
            image, label = process_image(image_path, square_size=self.image_square_size, augmentation=None)

        if self.verbose:
            print("Image: ", image_path)
            print("Label: ", label)

        # return image, label
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


if __name__ == '__main__':
    datagen = DataGenerator(train_or_test='train', augmentation=get_training_augmentation(SQUARE_SIZE, SQUARE_SIZE), verbose=True)
    for image, label in datagen:
        print(image.shape)
        print(label)

        plt.imshow(np.transpose(image, (1, 2, 0)))
        plt.show()
