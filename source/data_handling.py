import random
import albumentations as albu
import numpy as np
import os
import cv2
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

import config as cf
import utils

DATASETS_IN_USE = cf.DATASETS_IN_USE
TEST_DATASETS = cf.TEST_DATASETS
DATA_FOLDERS = cf.DATA_FOLDERS

SQUARE_SIZE = cf.SQUARE_SIZE
CLASS_LIST = list(cf.FULL_LABELS.values())


def get_training_augmentation(height, width, use_geometric_aug=True, use_colour_aug=False, prob_each_aug=0.75):
    def _get_training_augmentation(height, width):

        train_transform = [
            # colour based transforms
            # albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
            #
            # albu.IAAAdditiveGaussianNoise(p=prob_each_aug) if use_colour_aug else None,
            #
            # albu.CLAHE(p=prob_each_aug) if use_colour_aug else None,
            #
            # albu.RandomBrightness(p=prob_each_aug) if use_colour_aug else None,
            #
            # albu.RandomGamma(p=prob_each_aug) if use_colour_aug else None,
            #
            # albu.IAASharpen(p=prob_each_aug) if use_colour_aug else None,
            #
            # albu.Blur(blur_limit=3, p=prob_each_aug) if use_colour_aug else None,
            #
            # albu.MotionBlur(blur_limit=3, p=prob_each_aug) if use_colour_aug else None,
            #
            # albu.RandomContrast(p=prob_each_aug) if use_colour_aug else None,
            #
            # albu.HueSaturationValue(p=prob_each_aug) if use_colour_aug else None,
            #
            # albu.RandomBrightnessContrast(p=prob_each_aug) if use_colour_aug else None,
            #
            # albu.GaussNoise(p=prob_each_aug) if use_colour_aug else None,
            #
            # albu.MedianBlur(blur_limit=5, p=prob_each_aug) if use_colour_aug else None,
            #
            # albu.InvertImg(p=prob_each_aug) if use_colour_aug else None,
            #
            # albu.ToGray(p=prob_each_aug) if use_colour_aug else None,
            #
            # # geometric transforms
            albu.HorizontalFlip(p=prob_each_aug) if use_geometric_aug else None,
            #
            albu.VerticalFlip(p=prob_each_aug) if use_geometric_aug else None,
            #
            albu.Rotate(limit=180, p=prob_each_aug, border_mode=cv2.BORDER_CONSTANT) if use_geometric_aug else None,
            #
            # albu.IAAPerspective(p=prob_each_aug) if use_geometric_aug else None,
            #
            # albu.RandomCrop(height=height, width=width, always_apply=True, p=prob_each_aug) if use_geometric_aug else None,
            #
            # albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=0, shift_limit=0.0, p=prob_each_aug, border_mode=0) if use_geometric_aug else None,
            #
            # albu.RandomResizedCrop(height=height, width=width, scale=(0.5, 1.5), ratio=(0.75, 1.3333333333333333), interpolation=1, always_apply=True, p=prob_each_aug) if use_geometric_aug else None,
            #
            # albu.RGBShift(p=prob_each_aug) if use_geometric_aug else None,
            #
            # albu.ChannelShuffle(p=prob_each_aug) if use_geometric_aug else None,
            #
            # albu.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=prob_each_aug) if use_geometric_aug else None,
            #
            # albu.Cutout(num_holes=8, max_h_size=8, max_w_size=8, p=prob_each_aug) if use_geometric_aug else None,
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


def remove_black_borders(img, black_th=5):
    img_t = img.copy()
    img_t = cv2.cvtColor(img_t, cv2.COLOR_RGB2GRAY)

    # x axis
    centre_y = img_t.shape[1] // 2
    black_list = img_t[:, centre_y]
    black_list = np.where(black_list >= black_th)[0]

    if len(black_list) > 1:
        first_non_black = black_list[0]
        reverse_first_non_black = black_list[-1] + 1

        img = img[first_non_black:, :, :]
        img = img[:reverse_first_non_black - first_non_black, :, :]

    # y axis
    centre_x = img_t.shape[0] // 2
    black_list = img_t[centre_x, :]
    black_list = np.where(black_list >= black_th)[0]

    if len(black_list) > 1:
        first_non_black_y = black_list[0]
        reverse_first_non_black_y = black_list[-1] + 1

        img = img[:, first_non_black_y:, :]
        img = img[:, :reverse_first_non_black_y - first_non_black_y, :]

    return img


def basic_preprocessing(image, scale=SQUARE_SIZE):
    def scaleRadius(img, scale):
        x = img[img.shape[0] // 2, :, :].sum(1)
        r = (x > x.mean() / 10).sum() / 2
        s = scale * 1.0 / (r + 1e-8)

        return cv2.resize(img, (0, 0), fx=s, fy=s)

    a = image

    # scale img to a given radius
    a = scaleRadius(a, scale)

    # subtract local mean color
    a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128)

    # remove outer 10%
    b = np.zeros(a.shape)

    cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2), int(scale * 0.9), (1, 1, 1), -1, 8, 0)

    a = a * b + 128 * (1 - b)
    a = cv2.resize(a, (scale, scale))

    return a


def process_image(image_path, square_size, augmentation, return_label=True):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # remove black borders.
    # Doing this and resizing to a set size will help scale the eye to a similar size for all images
    image = remove_black_borders(image)

    # pad image to square
    height, width, _ = image.shape
    if height > width:
        pad = (height - width) // 2
        image = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_CONSTANT)
    elif width > height:
        pad = (width - height) // 2
        image = cv2.copyMakeBorder(image, pad, pad, 0, 0, cv2.BORDER_CONSTANT)

    image = cv2.resize(image, (square_size, square_size))

    # image = basic_preprocessing(image)

    if augmentation is not None:
        image = augmentation(image=image)['image']

    image = image / 255.0

    image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W) for pytorch

    if return_label:
        label = utils.get_label_from_path(image_path)
    else:
        return image

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
            class_list=CLASS_LIST,
            make_all_classes_equal=True,
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
        self.make_all_classes_equal = make_all_classes_equal
        self.per_class_cnt = {}
        self.org_per_class_cnt = {}
        self.class_list = class_list

        assert self.train_or_test in ['train', 'test'], "train_or_test must be either 'train' or 'test'"

        self.initialise_paths()

    def initialise_paths(self):
        # get train or test image paths from selected datasets of train or test
        self.def_image_paths = []
        if self.train_or_test == 'train':
            for dataset in self.train_datasets:
                paths = image_paths(dataset, self.train_or_test)
                self.def_image_paths.extend(paths)
        else:
            for dataset in self.test_datasets:
                paths = image_paths(dataset, self.train_or_test)
                self.def_image_paths.extend(paths)

        # shuffle paths if required
        if self.shuffle:
            random.shuffle(self.def_image_paths)

        # get length of paths and per class count
        self.image_paths_length = len(self.def_image_paths)
        self.per_class_cnt = get_class_wise_cnt(self.def_image_paths)
        self.org_per_class_cnt = self.per_class_cnt.copy()

        if self.make_all_classes_equal:
            self.max_class_len = max(self.org_per_class_cnt.values())
            self.max_iter_len = self.max_class_len * len(self.org_per_class_cnt)
            self.class_iter_indices = {}
            self.each_class_last_iter_index = {i: 0 for i in range(len(self.class_list))}
            self.equalised_paths = []  # smaller class paths are repeated to make all classes equal
            self.per_class_cnt = {i: self.max_class_len for i in range(len(self.class_list))}  # each class has equal count

            self.equalise_class_paths(shuffle=self.shuffle)

            self.image_paths_length = len(self.equalised_paths)

    def get_class_iter_indices(self):
        """
        Assign equal indices to each class so that each class is iterated over in each epoch

        Example:
            max_class_len = 3
            class_iter_indices = {
                0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2, 6: 0, 7: 1, 8: 2, 9: 0, 10: 1, 11: 2
            }
        """

        last_class_index = 0
        for i in range(self.max_iter_len):
            self.class_iter_indices[i] = last_class_index
            last_class_index += 1

            if last_class_index == len(self.org_per_class_cnt):
                last_class_index = 0

        # get class-wise paths
        self.class_paths = {i: [] for i in range(len(self.class_list))}
        for i, path in enumerate(self.def_image_paths):
            path_class = utils.get_label_from_path(path, return_one_hot=False)
            self.class_paths[path_class].append(path)

    def equalise_class_paths(self, shuffle=True):
        self.get_class_iter_indices()
        if shuffle:
            random.shuffle(self.def_image_paths)

        # change the def image paths to meet the lengths self.class_iter_indices indices
        classwise_iter_num = self.each_class_last_iter_index.copy()
        for i in range(self.max_iter_len):
            class_to_be_used = self.class_iter_indices[i]

            # if max samples is reached for the smaller classes, reset the class list and shuffle so that it can be added again to upsample the smaller classes
            if classwise_iter_num[class_to_be_used] >= len(self.class_paths[class_to_be_used]):
                classwise_iter_num[class_to_be_used] = 0
                if shuffle:
                    # we are shuffling here so that the last few samples of the class are not always the same. Doesn't hurt to shuffle
                    random.shuffle(self.class_paths[class_to_be_used])

            self.equalised_paths.append(self.class_paths[class_to_be_used][classwise_iter_num[class_to_be_used]])
            classwise_iter_num[class_to_be_used] += 1

        if shuffle:
            random.shuffle(self.equalised_paths)

        # update the def image paths to equalised paths
        self.def_image_paths = self.equalised_paths

        # audit to make sure the samples per class are equal
        audit_classwise_cnt = self.each_class_last_iter_index.copy()
        for path in self.def_image_paths:
            try:
                path_class = utils.get_label_from_path(path, return_one_hot=False)
            except ValueError:
                print(path)
                raise ValueError

            audit_classwise_cnt[path_class] += 1

        for class_idx in range(len(self.class_list)):
            if audit_classwise_cnt[class_idx] != self.max_class_len:
                raise f"Audit failed! Class {class_idx} has {audit_classwise_cnt[class_idx]} samples instead of {self.max_class_len}"

        print(f"Audit passed! All classes have {self.max_class_len} samples")

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
            prob_apply_augmentation=1.0,
            make_all_classes_equal=True
    ):
        self.train_or_test = train_or_test
        self.image_square_size = image_square_size
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.verbose = verbose
        self.prob_apply_augmentation = prob_apply_augmentation

        assert self.train_or_test in ['train', 'test'], "train_or_test must be either 'train' or 'test'"

        self.image_label_path_generator = CustomImagePathGenerator(
            train_or_test=self.train_or_test,
            shuffle=self.shuffle,
            verbose=self.verbose,
            make_all_classes_equal=make_all_classes_equal
        )
        self.per_class_cnt = self.image_label_path_generator.per_class_cnt

    def __len__(self):
        # return a lower length if num_workers is not 0, so that each worker gets a fair share of the data
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            return len(self.image_label_path_generator)

        else:
            per_worker = len(self.image_label_path_generator) // worker_info.num_workers
            residual = len(self.image_label_path_generator) % worker_info.num_workers
            if worker_info.id < residual:
                return per_worker + 1
            else:
                return per_worker

    def __getitem__(self, idx):
        actual_idx = idx

        try:
            image_path = self.image_label_path_generator.def_image_paths[actual_idx]
        except IndexError:
            print("IndexError: ", actual_idx)
            print("len(self.image_label_path_generator.def_image_paths): ", len(self.image_label_path_generator.def_image_paths))
            raise IndexError

        if self.prob_apply_augmentation >= random.random():
            image, label = process_image(image_path, square_size=self.image_square_size, augmentation=self.augmentation)
        else:
            image, label = process_image(image_path, square_size=self.image_square_size, augmentation=None)

        if self.verbose:
            print("Image: ", image_path)
            print("Label: ", label)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


if __name__ == '__main__':
    datagen = DataGenerator(train_or_test='train', augmentation=get_training_augmentation(SQUARE_SIZE, SQUARE_SIZE), verbose=True)
    for image, label in datagen:
        print(image.shape)
        print(label)

        plt.imshow(np.transpose(image, (1, 2, 0)))
        plt.show()
