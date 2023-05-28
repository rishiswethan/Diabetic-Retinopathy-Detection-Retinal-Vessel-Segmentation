import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import config as cf
import data_handling
import models
import utils

MODEL_SAVE_PATH = cf.MODEL_SAVE_PATH_BEST_VAL_LOSS
IMAGE_SIZE = cf.SQUARE_SIZE
FULL_LABELS = cf.FULL_LABELS
BEST_HP_JSON_SAVE_PATH = cf.BEST_HP_JSON_SAVE_PATH
NUM_CLASSES = cf.NUM_CLASSES


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
            labels=FULL_LABELS,
            best_hp_json_save_path=BEST_HP_JSON_SAVE_PATH,
            num_classes=NUM_CLASSES
            ):
        """
        Class to help initialise the model and encapsulate the predict function. Use this class to use the predict function encapsulated within it.

        Parameters:
        ----------
        model_save_path: str
            path to the model to load
        device: str, optional
            device to run the model on. If None, will use cuda by default if available, else will use cpu.
            Options: "cpu" or "cuda".
        batch_size: int, optional
            batch size to use when predicting. Reduce if running out of memory when using cuda
        image_size: int, optional
            image size to use when predicting. Do not change unless you trained the model with a different image size
        verbose: bool, optional
            whether to print out information when predicting
        labels: list, optional
            list of labels to use when predicting. Do not change unless you trained the model with different labels
        best_hp_json_save_path: str, optional
            path to the json file containing the best hyperparameters from the tuning process
        num_classes: int, optional
            number of classes to use when predicting. Do not change unless you trained the model with a different number of classes

        """

        self.model_save_path = model_save_path

        # load best hyperparameters
        best_hp = utils.load_dict_from_json(best_hp_json_save_path)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = models.models_dict[best_hp["conv_model"]](num_classes=num_classes, class_weights=None)
        self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))

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
        Predict the condition of the retina for the given image(s)

        Parameters
        ----------
        images: str or list of str
            Argument can be one of the following:
                - string path to a single image
                - string path to a folder containing images
                - list of string paths to images

        Returns
        -------
        outputs: dict of str: int
            You'll get a dictionary of image paths and their corresponding predicted labels.

            Example:
                >>> Predict().predict(cf.INPUT_FOLDER)  # contains 3 images

                Output:
                    {'<cf.INPUT_FOLDER>/test_IDRiD_016-Severe-3.jpg': 'Severe',
                     '<cf.INPUT_FOLDER>/test_IDRiD_047-No_DR-0.jpg': 'No_DR',
                     '<cf.INPUT_FOLDER>/train_IDRiD_236-Moderate-2.jpg': 'Moderate'}

        """

        if (type(images) is str):
            if os.path.isdir(images):
                # Get all images in the folder
                # images = [os.path.join(images, img) for img in os.listdir(images)]
                images_list = []
                for img in os.listdir(images):
                    file_path = os.path.join(images, img)
                    if os.path.isfile(file_path):
                        images_list.append(file_path)

                images = images_list


            else:
                # Single image
                images = [images]

        elif type(images) is not list:
            # if it's not a list of images as well as not a string, then it's not valid

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

        # Convert to numerical labels from softmax outputs
        simple_outputs = np.argmax(outputs, axis=1)

        output_dict = {}
        for i in range(len(images)):
            output_label = self.labels[simple_outputs[i]]

            output_dict[images[i]] = output_label

            if self.verbose:
                img_transp = preprocessed_images[i].transpose(1, 2, 0)

                plt.imshow(img_transp)
                plt.title("Prediction: " + output_label)
                plt.show()

        return output_dict


if __name__ == '__main__':
    # Example usage:

    # Initialise the class
    predict = Predict(verbose=True)
    pred = predict.predict("/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/FundusImages/data/org_data/MESSIDOR/archives/Base14/20060530_53702_0100_PP.tif")
    print(pred)

