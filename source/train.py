from torchvision.models import *
from torch.nn import functional as F
from torch import nn
import torch

import pytorch_utils.callbacks as pt_callbacks
import pytorch_utils.training_utils as pt_train
import source.data_handling as data_handling
import source.config as cf
import source.utils as utils

DEVICE = cf.DEVICE
NUM_CLASSES = cf.NUM_CLASSES
INITIAL_EPOCH = cf.INITIAL_EPOCH
INITIAL_LR = cf.INITIAL_LR
MAX_THREADS = cf.MAX_THREADS

SQUARE_SIZE = cf.SQUARE_SIZE

MODEL_SAVE_PATH_BEST_VAL_LOSS = cf.MODEL_SAVE_PATH_BEST_VAL_LOSS
VAL_CALLBACK_OBJ_PATH = cf.VAL_CALLBACK_OBJ_PATH
MODEL_SAVE_PATH_BEST_TRAIN_LOSS = cf.MODEL_SAVE_PATH_BEST_TRAIN_LOSS
TRAIN_CALLBACK_OBJ_PATH = cf.TRAIN_CALLBACK_OBJ_PATH


def get_data_generators(height=SQUARE_SIZE, width=SQUARE_SIZE, train_shuffle=True, val_shuffle=True):
    train_gen = data_handling.DataGenerator(
        'train',
        augmentation=data_handling.get_training_augmentation(height=height, width=width),
        shuffle=train_shuffle,
        verbose=False
    )

    val_gen = data_handling.DataGenerator(
        'test',
        augmentation=None,
        shuffle=val_shuffle,
        verbose=False
    )

    return train_gen, val_gen


def get_callbacks(
        optimiser,
        result,
        model,
        defined_callbacks=None,
        continue_training=False,
        other_stats=None
):

    if defined_callbacks is None:
        defined_callbacks = {
            'val': pt_callbacks.Callbacks(optimizer=optimiser,
                                          model_save_path=MODEL_SAVE_PATH_BEST_VAL_LOSS,
                                          training_stats_path=VAL_CALLBACK_OBJ_PATH,
                                          continue_training=continue_training),
            'train': pt_callbacks.Callbacks(optimizer=optimiser,
                                            model_save_path=MODEL_SAVE_PATH_BEST_TRAIN_LOSS,
                                            training_stats_path=TRAIN_CALLBACK_OBJ_PATH,
                                            continue_training=continue_training)
        }

    defined_callbacks['val'].reduce_lr_on_plateau(
        monitor_value=result["val_loss"],
        mode='max',
        factor=cf.REDUCE_LR_FACTOR_VAL,
        patience=cf.REDUCE_LR_PATIENCE_VAL,
        indicator_text="Val LR scheduler: "
    )
    defined_callbacks['train'].reduce_lr_on_plateau(
        monitor_value=result["train_loss"],
        mode='max',
        factor=cf.REDUCE_LR_FACTOR_TRAIN,
        patience=cf.REDUCE_LR_PATIENCE_TRAIN,
        indicator_text="Train LR scheduler: "
    )
    defined_callbacks['train'].model_checkpoint(
        model=model,
        monitor_value=result["train_loss"],
        mode='max',
        other_stats=other_stats,
        indicator_text="Train checkpoint: "
    )
    defined_callbacks['val'].model_checkpoint(
        model=model,
        monitor_value=result["val_loss"],
        mode='max',
        other_stats=other_stats,
        indicator_text="Val checkpoint: "
    )
    stop_flag = defined_callbacks['val'].early_stopping(
        monitor_value=result[cf.EARLY_STOPPING_MONITOR],
        mode='min',
        patience=cf.EARLY_STOPPING_PATIENCE,
        indicator_text="Early stopping: "
    )
    defined_callbacks['train'].clear_memory()
    print("_________")

    return defined_callbacks, stop_flag


class CustomModelBase(pt_train.CustomModelBase):
    """
    ModelBase override for training and validation steps
    """
    def __init__(self, class_weights):
        super(CustomModelBase, self).__init__()
        self.class_weights = class_weights

    def training_step(self, batch):
        # print("batch: ", len(batch))
        images, labels = batch

        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels, weight=self.class_weights)  # Calculate loss with class weights
        acc = pt_train._accuracy(out, labels)  # Calculate accuracy
        return loss, acc

    def validation_step(self, batch):
        images, labels = batch

        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels, weight=self.class_weights)  # Calculate loss with class weights
        acc = pt_train._accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}


class CustomModel(CustomModelBase):
    """
        A custom model that inherits from CustomModelBase.
        This class is meant to be used with the training_utils.py module.

        Parameters
        ----------
        input_shape - The shape of the input data (n, 3, height, width), (n, 1404)
        dropout_rate - The dropout rate to use
        dense_units - The number of units in the dense layer
        num_layers - The number of dense layers
        l1_l2_reg - The L1 and L2 regularization to use (Not implemented yet)
        layers_batch_norm - Whether to use batch normalization in the dense layers
        conv_model_name - The name of the convolutional model to use. Choose from the list in the get_conv_model function
        class_weights : list - The class weights to use. If None, all classes will have the same weight
        device - The device to use
    """

    def __init__(
            self,
            conv_model_name,
            input_shape,
            class_weights=None,
            device=DEVICE,
    ):

        if class_weights is None:
            class_weights = torch.ones(NUM_CLASSES)
        else:
            class_weights = torch.tensor(class_weights)

        # convert to cuda tensor
        class_weights = class_weights.to(device)

        super(CustomModel, self).__init__(class_weights=class_weights)

        self.base_model_conv = self.get_conv_model(conv_model_name)

        # Remove the final classification layer (fc)
        self.base_model = nn.Sequential(*list(self.base_model_conv.children())[:-1])

        self.flatten = nn.Flatten()

        # Determine the output size of the base model
        with torch.no_grad():
            sample_input = torch.randn(1, input_shape[0], input_shape[1], input_shape[2])
            print("sample_input: ", sample_input.shape)
            self.base_output_size = self.base_model(sample_input).numel()

        self.out_no_lands = nn.Linear(self.base_output_size, NUM_CLASSES)

    def get_conv_model(self, conv_model_name):
        if conv_model_name == "resnet50":
            return resnet50(weights=ResNet50_Weights.DEFAULT)
        elif conv_model_name == "resnet18":
            return resnet18(weights=ResNet18_Weights.DEFAULT)
        elif conv_model_name == "resnet34":
            return resnet34(weights=ResNet34_Weights.DEFAULT)
        elif conv_model_name == "resnet101":
            return resnet101(weights=ResNet101_Weights.DEFAULT)
        elif conv_model_name == "resnet152":
            return resnet152(weights=ResNet152_Weights.DEFAULT)
        elif conv_model_name == "resnext50_32x4d":
            return resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        elif conv_model_name == "resnext101_32x8d":
            return resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.DEFAULT)
        elif conv_model_name == "wide_resnet50_2":
            return wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        elif conv_model_name == "wide_resnet101_2":
            return wide_resnet101_2(weights=Wide_ResNet101_2_Weights.DEFAULT)
        elif conv_model_name == "inception":
            return inception_v3(weights=Inception_V3_Weights.DEFAULT)
        elif conv_model_name == "googlenet":
            return googlenet(weights=GoogLeNet_Weights.DEFAULT)
        elif conv_model_name == "mobilenet":
            return mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        elif conv_model_name == "densenet":
            return densenet121(weights=DenseNet121_Weights.DEFAULT)
        elif conv_model_name == "alexnet":
            return alexnet(weights=AlexNet_Weights.DEFAULT)
        elif conv_model_name == "vgg16":
            return vgg16(weights=VGG16_Weights.DEFAULT)
        elif conv_model_name == "squeezenet":
            return squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
        elif conv_model_name == "shufflenet":
            return shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
        elif conv_model_name == "mnasnet":
            return mnasnet1_0(weights=MNASNet1_0_Weights.DEFAULT)
        else:
            raise ValueError("Invalid model name, exiting...")

    def forward(self, x):
        x = self.base_model(x)
        x = self.flatten(x)
        x = self.out_no_lands(x)

        return x


def train(hp_dict, metric='val_acc', metric_mode='max', preprocess_again=False, initial_lr=INITIAL_LR, epochs=INITIAL_EPOCH, max_threads=MAX_THREADS):
    """
    Once the best hyperparameters are found using tune_hyperparameters(), call this function to train the model with the best hyperparameters found.

    Parameters
    ----------
    hp_dict: dict
        Contains the hyperparameters to be used for training and preprocessing.

    metric: str
        Target metric whose max or min value is to be found in the training process and returned. Will be used to find the best hyperparameters.

    metric_mode: str
        'max' or 'min' depending on whether the metric is to be maximised or minimised

    preprocess_again: bool
        If True, the data will be preprocessed again. If False, the data will be loaded from the preprocessed files.

    initial_lr: float
        Initial learning rate to be used for training. Can be scheduled to change during training using the reduce_lr_on_plateau callback in the pytorch_callbacks.py file.

    epochs: int
        Number of epochs to train for. Can step out of the training loop early if the early_stopping callback in the pytorch_callbacks.py file is triggered.

    Returns
    -------
    opt_result: float
        The best value of the metric found during training. This is the value that will be used to find the best hyperparameters.

    """
    def get_min_max_vale(history, key):
        min = 99999
        max = -99999
        for i in range(len(history)):
            if history[i][key] < min:
                min = history[i][key]
            if history[i][key] > max:
                max = history[i][key]

        return min, max

    # Train hyperparameters
    batch_size = hp_dict['batch_size']
    conv_model = hp_dict['conv_model']

    # get and set generators
    train_gen, val_gen = get_data_generators()
    train_loader = torch.utils.data.DataLoader(train_gen, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_gen, batch_size=batch_size, shuffle=True, num_workers=0)

    # Create the model
    input_shape = (3, SQUARE_SIZE, SQUARE_SIZE)
    print("Class cnt: ", train_gen.per_class_cnt)
    class_weights = utils.get_class_weights(train_gen.per_class_cnt)
    print("Class weights: ", class_weights)

    model = CustomModel(
        conv_model_name=conv_model,
        input_shape=input_shape,
        class_weights=class_weights,
    )

    # Train the model using torch
    history = pt_train.fit(
        epochs=epochs,
        lr=initial_lr,
        model=model,
        callbacks_function=get_callbacks,
        train_loader=train_loader,
        val_loader=val_loader
    )

    if metric and metric_mode:
        acc_min, acc_max = get_min_max_vale(history, metric)
        opt_result = acc_min if metric_mode == 'min' else acc_max

        # set to - if metric_mode is min, else set to +. This is for hyperopt to work
        opt_result = -opt_result if metric_mode == 'min' else opt_result

        return opt_result


if __name__ == "__main__":
    # tune_hyperparameters()
    best_hp_dict = {
        'batch_size': 16,
        'conv_model': 'resnet101',
    }
    train(hp_dict=best_hp_dict, metric='val_acc', metric_mode='max', preprocess_again=True, initial_lr=INITIAL_LR, epochs=INITIAL_EPOCH, max_threads=MAX_THREADS)
