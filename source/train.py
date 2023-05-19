import numpy as np
from torchvision.models import *
from torch.nn import functional as F
from torch import nn
import torch
import torch_inception_resnet_v2.model as inception_resnet_v2
import matplotlib.pyplot as plt

import pytorch_utils.callbacks as pt_callbacks
import pytorch_utils.training_utils as pt_train
import source.data_handling as data_handling
import source.config as cf
import source.utils as utils
import source.models as models

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

FULL_LABELS = cf.FULL_LABELS


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
        monitor_value=result["val_acc"],
        mode='max',
        factor=cf.REDUCE_LR_FACTOR_VAL,
        patience=cf.REDUCE_LR_PATIENCE_VAL,
        indicator_text="Val LR scheduler: "
    )
    defined_callbacks['train'].reduce_lr_on_plateau(
        monitor_value=result["train_loss"],
        mode='min',
        factor=cf.REDUCE_LR_FACTOR_TRAIN,
        patience=cf.REDUCE_LR_PATIENCE_TRAIN,
        indicator_text="Train LR scheduler: "
    )
    defined_callbacks['train'].model_checkpoint(
        model=model,
        monitor_value=result["train_loss"],
        mode='min',
        other_stats=other_stats,
        indicator_text="Train checkpoint: "
    )
    defined_callbacks['val'].model_checkpoint(
        model=model,
        monitor_value=result["val_acc"],
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
    print("Training hyperparameters: ", hp_dict)
    batch_size = hp_dict['batch_size']
    conv_model = hp_dict['conv_model']

    # get and set generators
    train_gen, val_gen = get_data_generators()
    train_loader = torch.utils.data.DataLoader(train_gen, batch_size=batch_size, shuffle=True, num_workers=MAX_THREADS)
    val_loader = torch.utils.data.DataLoader(val_gen, batch_size=batch_size, shuffle=True, num_workers=MAX_THREADS)

    # Create the model
    input_shape = (3, SQUARE_SIZE, SQUARE_SIZE)
    print("Class cnt: ", train_gen.per_class_cnt)
    class_weights = utils.get_class_weights(train_gen.per_class_cnt)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    print("Class weights: ", class_weights)

    model = models.EfficientNetB3(num_classes=NUM_CLASSES, class_weights=class_weights)
    # model = models.ViT_B_16(num_classes=NUM_CLASSES, class_weights=class_weights)

    # visualise training set and model
    visualise_generator(hp_dict, train_loader, num_images=3)

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


def visualise_generator(hp_dict, data_loader, num_classes=NUM_CLASSES, full_labels=FULL_LABELS, num_images=None):
    if type(data_loader) == str:
        if data_loader == 'train':
            data_generator = get_data_generators()[0]
        elif data_loader == 'val':
            data_generator = get_data_generators()[1]
    elif type(data_loader) != torch.utils.data.DataLoader:
        raise TypeError("data_loader must be of type str or torch.utils.data.DataLoader, or \"train\" or \"val\"")

    if type(data_loader) != torch.utils.data.DataLoader:
        data_loader = torch.utils.data.DataLoader(data_generator, batch_size=1, shuffle=False, num_workers=10)

    model = models.EfficientNetB3(num_classes=num_classes).to(DEVICE)
    model.eval()

    cnt = 0
    for batch in data_loader:
        for (image, label) in zip(batch[0], batch[1]):

            print("\nImage: ", image.shape)
            print("Label: ", label)

            # get prediction
            image = image.unsqueeze(0).to(DEVICE)
            pred = model(image)
            print("NN output: ", pred)
            pred = int(torch.argmax(pred, dim=1).detach().cpu().numpy())
            print("Prediction: ", pred, "Pred name: ", full_labels[pred])

            image = image.squeeze(0).cpu().numpy()
            image = np.transpose(image, (1, 2, 0))

            label = int(torch.argmax(label, dim=0).detach().cpu().numpy())
            print("Label: ", label, "Label name: ", full_labels[label], "\n")

            plt.imshow(image)
            plt.title("Label name: " + str(full_labels[label]) + " | Pred name: " + str(full_labels[pred]))
            plt.show()

            cnt += 1
            if num_images and cnt >= num_images:
                return


if __name__ == "__main__":
    # tune_hyperparameters()
    best_hp_dict = {
        'batch_size': 8,
        'conv_model': 'vit',
    }

    train(hp_dict=best_hp_dict, metric='val_acc', metric_mode='max', preprocess_again=True, initial_lr=INITIAL_LR, epochs=INITIAL_EPOCH, max_threads=MAX_THREADS)

    # visualise_generator(hp_dict=best_hp_dict, generator='val')
