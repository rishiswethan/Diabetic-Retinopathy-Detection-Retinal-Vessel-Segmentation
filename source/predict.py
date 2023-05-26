import os
import numpy as np
import torch
import cv2
import time
import matplotlib.pyplot as plt

import source.config as cf
import source.data_handling as data_handling

MODEL_SAVE_PATH = cf.MODEL_SAVE_PATH_BEST_VAL_LOSS
IMAGE_SIZE = cf.SQUARE_SIZE
FULL_LABELS = cf.FULL_LABELS


def _batch(iterable, n=1):
    """
    Helper function to batch an iterable into batches of size n
    """

    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class Predict:
    def __init__(
            self,
            model_save_path=MODEL_SAVE_PATH,
            device=None,
            batch_size=8,
            image_size=IMAGE_SIZE,
            verbose=False,
            labels=FULL_LABELS
            ):
        """
        Class to help initialise the model and encapsulate the predict function

        :param model_save_path (str): path to the model to load
        :param device (str) (optional): device to run the model on. If None, will use cuda by default if available, else will use cpu.
            Options: "cpu" or "cuda".
        :param batch_size (int) (optional): batch size to use when predicting. Reduce if running out of memory when using cuda
        :param image_size (int) (optional): image size to use when predicting. Do not change unless you trained the model with a different image size
        """

        self.model_save_path = model_save_path

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = torch.load(self.model_save_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

        self.batch_size = batch_size
        self.image_size = image_size
        self.verbose = verbose
        self.labels = labels

        if self.verbose:
            print("Initialised Predict class with model_save_path: {}".format(self.model_save_path))
            print("Using device: {}; Batch size: {}".format(self.device, self.batch_size))
            print("Now ready to predict using the predict function")

    def predict(self, images):
        """
        Predict the similarity between images_1 and images_2. Image_1 is typically the folded image and image_2 is the spread image

        :param images_1 str or list of str: path to the image or list of paths to the images
        :param images_2 str or list of str: path to the image or list of paths to the images
        :return: list of ints: list of binary predictions (0 or 1) for each image pair
        """

        if (type(images) is str):
            images = [images]

        elif type(images) is not list:
            raise TypeError("images must be a list of image paths or a single image path")

        outputs = []
        preprocessed_images = []
        for batch_images in _batch(images, self.batch_size):
            tensor_images = []

            for img in batch_images:
                # Load and preprocess the image
                image = data_handling.process_image(img, square_size=self.image_size, augmentation=None, return_label=False)
                preprocessed_images.append(image)

                tensor_images.append(image)

            # Stack images into a single tensor and convert to torch tensor
            tensor_images = torch.tensor(np.stack(tensor_images), dtype=torch.float32).to(self.device)

            # Predict
            with torch.no_grad():
                output = self.model(tensor_images)
                output = output.cpu().numpy()
                outputs.append(output)

        # Concatenate outputs from various batches into a simple list of outputs
        outputs = np.concatenate(outputs, axis=0)

        # Convert to binary predictions from [0, 1] to 0 or 1
        simple_outputs = np.argmax(outputs, axis=1)

        if self.verbose:
            for i in range(len(images)):
                output_label = self.labels[simple_outputs[i]]

                plt.imshow(preprocessed_images[i])
                plt.title("Prediction: " + output_label)
                plt.show()

        return simple_outputs


if __name__ == '__main__':
    # Example usage:

    # Initialise the class
    predict = Predict(verbose=True)
    pred = predict.predict("/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/FundusImages/data/org_data/MESSIDOR/archives/Base14/20060530_53702_0100_PP.tif")
    print(pred)

