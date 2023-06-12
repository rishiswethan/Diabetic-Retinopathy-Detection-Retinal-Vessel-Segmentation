import os
import shutil

import numpy as np
import torch

import source_segment.segmentation_tools.utils as seg_utils
import source_segment.segmentation_tools.segmentation_config as seg_cf
import source_segment.config as cf
import source_segment.segmentation_tools.data_handling as data_handling
import source_segment.segmentation_tools.train as train


MODEL_SAVE_PATH = cf.MODEL_SAVE_PATH
IMAGE_SIZE = seg_cf.HEIGHT


class Predict:
    def __init__(
            self,
            model_save_path=MODEL_SAVE_PATH,
            device=None,
            batch_size=8,
            image_size=IMAGE_SIZE,
            verbose=False
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

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = train.get_model_def()
        self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))

        self.model.eval()
        self.model.to(self.device)

        self.batch_size = batch_size
        self.image_size = image_size
        self.verbose = verbose

        if self.verbose:
            print("Initialised Predict class with model_save_path: {}".format(self.model_save_path))
            print("Using device: {}; Batch size: {}".format(self.device, self.batch_size))
            print("Now ready to predict using the predict function")

    def _batch(self, iterable, n=1):
        """
        Helper function to batch an iterable into batches of size n
        """

        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def predict(self, images, return_actual_preds=False):
        """
        Predict the condition of the retina for the given image(s)

        Parameters
        ----------
        images: str or list of str
            Argument can be one of the following:
                - string path to a single image
                - string path to a folder containing images
                - list of string paths to images
        return_actual_preds: bool, optional
            whether to return the actual predictions (softmax probabilities) or the predicted labels (colour names).
            Once you initialise the Predict class, you can use the labels attribute to see the order of the labels.
            Example:
                >>> predict = Predict()
                >>> predict.predict(cf.INPUT_FOLDER, return_actual_preds=True)  # return 3 masks
        Returns
        -------
        outputs: dict of str: int
            You'll get a dictionary of image paths and their corresponding predicted labels.

            Example:
                >>> Predict().predict(cf.INPUT_FOLDER, return_actual_preds=False)  # contains 3 images

                Output:
                    {'<cf.INPUT_FOLDER>/image_1.jpg': <mask numpy array>,
                     '<cf.INPUT_FOLDER>/image_2.jpg': <mask numpy array>,
                     '<cf.INPUT_FOLDER>/image_3.jpg': <mask numpy array>}

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

        print("\n\nPredicting...\n")

        outputs = []
        preprocessed_images = []
        img_cnt = 1
        for batch_images in self._batch(images, self.batch_size):
            tensor_images = []

            for img in batch_images:
                # Load and preprocess the image
                image, _ = data_handling.read_fix_img_mask(img=img, mask=None)
                image = data_handling.preprocess_image_mask(image=image, mask=None)["image"]

                # covert the image to pytorch format
                image = image.transpose(2, 0, 1)

                print(f"Image {img_cnt}: ", image.shape)
                img_cnt += 1

                preprocessed_images.append(image)

                tensor_images.append(image)

            # Stack images into a single tensor and convert to torch tensor
            tensor_images = torch.tensor(np.stack(tensor_images), dtype=torch.float32).to(self.device)

            # Predict
            with torch.no_grad():
                output = self.model(tensor_images)
                output = output.cpu().numpy()
                print("output: ", output.shape)
                outputs.append(output)

        # Concatenate outputs from various batches into a simple list of outputs
        outputs = np.concatenate(outputs, axis=0)

        if return_actual_preds:
            return outputs

        output_dict = {}
        for i in range(len(images)):
            output_mask = outputs[i]
            print("--output_mask: ", output_mask.shape, np.unique(output_mask))
            output_mask = np.argmax(output_mask, axis=0)
            print("output_label: ", output_mask.shape, np.unique(output_mask))

            output_dict[images[i]] = output_mask

            if self.verbose:
                img_transp = preprocessed_images[i].transpose(1, 2, 0)

                seg_utils.display([img_transp, output_dict[images[i]]],
                                  ["image", "mask"])

        return output_dict


if __name__ == '__main__':
    Predict(verbose=True).predict(cf.INPUT_FOLDER, return_actual_preds=False)
