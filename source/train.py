import gc
import os


import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from hyperopt import fmin, space_eval, Trials

import pytorch_utils.callbacks as pt_callbacks
import pytorch_utils.training_utils as pt_train
import pytorch_utils.hyper_tuner as pt_tuner

import data_handling
import config as cf
import utils
import models

#############################################################################
# define constants
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

TUNE_HP_RANGES = cf.TUNE_HP_RANGES
BEST_HP_JSON_SAVE_PATH = cf.BEST_HP_JSON_SAVE_PATH
TUNER_CSV_SAVE_PATH = cf.TUNER_CSV_SAVE_PATH
TUNER_SAVE_PATH = cf.TUNER_SAVE_PATH
MAX_TRIALS = cf.MAX_TRIALS

REDUCE_LR_FACTOR_VAL = cf.REDUCE_LR_FACTOR_VAL
REDUCE_LR_PATIENCE_VAL = cf.REDUCE_LR_PATIENCE_VAL
REDUCE_LR_FACTOR_TRAIN = cf.REDUCE_LR_FACTOR_TRAIN
REDUCE_LR_PATIENCE_TRAIN = cf.REDUCE_LR_PATIENCE_TRAIN
EARLY_STOPPING_PATIENCE = cf.EARLY_STOPPING_PATIENCE
TUNING_EARLY_STOPPING_PATIENCE = cf.TUNING_EARLY_STOPPING_PATIENCE
EARLY_STOPPING_MONITOR = cf.EARLY_STOPPING_MONITOR
EARLY_STOPPING_MONITOR_MODE = cf.EARLY_STOPPING_MONITOR_MODE
TRAIN_TUNE_TARGET = cf.TRAIN_TUNE_TARGET
TRAIN_TUNE_MODE = cf.TRAIN_TUNE_MODE

FULL_LABELS = cf.FULL_LABELS
#############################################################################


def get_data_generators(
        height=SQUARE_SIZE,
        width=SQUARE_SIZE,
        prob_apply_augmentation=1.0,
        use_geometric_augmentation=True,
        use_colour_augmentation=True,
        train_shuffle=True,
        val_shuffle=True,
        make_all_classes_equal=True,
):
    train_gen = data_handling.DataGenerator(
        'train',
        augmentation=data_handling.get_training_augmentation(height=height, width=width, use_geometric_aug=use_geometric_augmentation, use_colour_aug=use_colour_augmentation),
        # augmentation=None,
        shuffle=train_shuffle,
        prob_apply_augmentation=prob_apply_augmentation,
        verbose=False,
        make_all_classes_equal=make_all_classes_equal
    )

    val_gen = data_handling.DataGenerator(
        'test',
        augmentation=None,
        shuffle=val_shuffle,
        verbose=False,
        make_all_classes_equal=False
    )

    return train_gen, val_gen


def get_callbacks(
       reduce_lr_factor_val=REDUCE_LR_FACTOR_VAL,
       reduce_lr_patience_val=REDUCE_LR_PATIENCE_VAL,
       reduce_lr_factor_train=REDUCE_LR_FACTOR_TRAIN,
       reduce_lr_patience_train=REDUCE_LR_PATIENCE_TRAIN,
       early_stopping_patience=EARLY_STOPPING_PATIENCE,
):
    def _get_callbacks(
            optimiser,
            result,
            model,
            defined_callbacks=None,
            continue_training=False,
            other_stats=None,
            model_save_path_best_val_loss=MODEL_SAVE_PATH_BEST_VAL_LOSS,
            training_stats_path_val=VAL_CALLBACK_OBJ_PATH,
            model_save_path_best_train_loss=MODEL_SAVE_PATH_BEST_TRAIN_LOSS,
            training_stats_path_train=TRAIN_CALLBACK_OBJ_PATH,
            early_stopping_monitor=EARLY_STOPPING_MONITOR,
            early_stopping_monitor_mode=EARLY_STOPPING_MONITOR_MODE
    ):

            if defined_callbacks is None:
                defined_callbacks = {
                    'val': pt_callbacks.Callbacks(optimizer=optimiser,
                                                  model_save_path=model_save_path_best_val_loss,
                                                  training_stats_path=training_stats_path_val,
                                                  continue_training=continue_training),

                    'train': pt_callbacks.Callbacks(optimizer=optimiser,
                                                    model_save_path=model_save_path_best_train_loss,
                                                    training_stats_path=training_stats_path_train,
                                                    continue_training=continue_training)
                }

            defined_callbacks['val'].reduce_lr_on_plateau(
                monitor_value=result["val_acc"],
                mode='max',
                factor=reduce_lr_factor_val,
                patience=reduce_lr_patience_val,
                indicator_text="Val LR scheduler: "
            )
            defined_callbacks['train'].reduce_lr_on_plateau(
                monitor_value=result["train_acc"],
                mode='max',
                factor=reduce_lr_factor_train,
                patience=reduce_lr_patience_train,
                indicator_text="Train LR scheduler: "
            )
            defined_callbacks['train'].model_checkpoint(
                model=model,
                monitor_value=result["train_acc"],
                mode='max',
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
                monitor_value=result[early_stopping_monitor],
                mode=early_stopping_monitor_mode,
                patience=early_stopping_patience,
                indicator_text="Early stopping: "
            )
            defined_callbacks['train'].clear_memory()
            print("_________")

            return defined_callbacks, stop_flag

    return _get_callbacks


def train(
        hp_dict,
        metric=TRAIN_TUNE_TARGET,
        metric_mode=TRAIN_TUNE_MODE,
        initial_lr=INITIAL_LR,
        epochs=INITIAL_EPOCH,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        num_classes=NUM_CLASSES,
        device_name=DEVICE,
        initial_visualise=False,
        continue_training=True,
        fine_tune=False,
        model_save_path=MODEL_SAVE_PATH_BEST_TRAIN_LOSS,
):
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
    initial_lr: float
        Initial learning rate to be used for training. Can be scheduled to change during training using the reduce_lr_on_plateau callback in the pytorch_callbacks.py file.
    epochs: int
        Number of epochs to train for. Can step out of the training loop early if the early_stopping callback in the pytorch_callbacks.py file is triggered.
    initial_visualise: bool
        Whether to visualise some training examples before training begins.
    fine_tune (optional): bool or str
        Path of the pretrained model to be loaded for fine tuning. If False, no pretrained model will be loaded. Default is None.

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

    # Clear memory before training
    torch.cuda.empty_cache()
    gc.collect()

    # Train hyperparameters
    print("Training hyperparameters: ", hp_dict)
    batch_size = hp_dict['batch_size']
    conv_model = hp_dict['conv_model']
    prob_apply_augmentation = hp_dict['prob_apply_augmentation']
    use_geometric_augmentation = hp_dict['use_geometric_augmentation']
    use_colour_augmentation = hp_dict['use_colour_augmentation']

    # get callbacks based on given hyperparameters
    callbacks_func = get_callbacks(
        reduce_lr_factor_val=hp_dict['reduce_lr_factor_val'],
        reduce_lr_patience_val=hp_dict['reduce_lr_patience_val'],
        reduce_lr_factor_train=hp_dict['reduce_lr_factor_train'],
        reduce_lr_patience_train=hp_dict['reduce_lr_patience_train'],
        early_stopping_patience=early_stopping_patience,
    )

    # get and set generators
    train_gen, val_gen = get_data_generators(prob_apply_augmentation=prob_apply_augmentation,
                                             use_geometric_augmentation=use_geometric_augmentation,
                                             use_colour_augmentation=use_colour_augmentation)

    train_loader = torch.utils.data.DataLoader(train_gen, batch_size=batch_size, shuffle=True, num_workers=MAX_THREADS)
    val_loader = torch.utils.data.DataLoader(val_gen, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create the model
    print("Class cnt: ", train_gen.per_class_cnt)
    class_weights = utils.get_class_weights(train_gen.per_class_cnt)
    class_weights = torch.FloatTensor(class_weights).to(device_name)
    print("Class weights: ", class_weights)
    print("Num train images: ", len(train_gen))
    print("Num val images: ", len(val_gen))

    model = models.models_dict[conv_model](num_classes=num_classes, class_weights=class_weights)

    if continue_training or fine_tune:
        model_save_path = fine_tune if fine_tune else model_save_path
        print("Loading model from: ", model_save_path)

        model.load_state_dict(torch.load(model_save_path, map_location=device_name))

    model.to(device_name)

    # visualise training set and model
    if initial_visualise:
        visualise_generator(train_loader, num_images=5, model=model, run_evaluation=False, val_batch_size=batch_size)

    print("_________________________________________________________________________________________________")
    print("Training model: ", conv_model, "\n")
    model.train()

    # Train the model using torch
    history = pt_train.fit(
        epochs=epochs,
        lr=initial_lr,
        model=model,
        callbacks_function=callbacks_func,
        train_loader=train_loader,
        val_loader=val_loader,
        continue_training=continue_training,
    )

    if metric and metric_mode:
        acc_min, acc_max = get_min_max_vale(history, metric)
        opt_result = acc_min if metric_mode == 'min' else acc_max

        # set to - if metric_mode is min, else set to +. This is for hyperopt to work
        opt_result = -opt_result if metric_mode == 'min' else opt_result

        return opt_result


def train_using_best_hp(best_hp_json_save_path=BEST_HP_JSON_SAVE_PATH,
                        early_stopping_patience=EARLY_STOPPING_PATIENCE,
                        continue_training=False,
                        fine_tune=None):
    """
    Train the model using the best hyperparameters found using hyperopt
    """

    # load best hyperparameters
    best_hp = utils.load_dict_from_json(best_hp_json_save_path)

    # train using the best hyperparameters
    train(best_hp, initial_visualise=True, early_stopping_patience=early_stopping_patience, continue_training=continue_training, fine_tune=fine_tune)


def train_to_tune(hp_dict,
                  tuning_early_stopping_patience=TUNING_EARLY_STOPPING_PATIENCE):
    """
    Train the model using the given hyperparameters. Used for tuning the hyperparameters.
    """

    # train using the given hyperparameters
    return train(hp_dict, initial_visualise=False, early_stopping_patience=tuning_early_stopping_patience)


def hyper_parameter_optimise(
        search_space=TUNE_HP_RANGES,
        best_hp_json_save_path=BEST_HP_JSON_SAVE_PATH,
        tuner_csv_save_path=TUNER_CSV_SAVE_PATH,
        tuner_obj_save_path=TUNER_SAVE_PATH,
        tune_target=TRAIN_TUNE_TARGET,
        max_trials=MAX_TRIALS,
        load_if_exists=True,
):
    """
    Main function for hyperparameter optimisation using hyperopt

    Parameters
    ----------
    search_space: dict
        Example:
            tune_hp_ranges = {
                "dropout_rate": ([0.0, 0.3, 4], 'range')
                "conv_model": (["resnet18", "resnet101", "resnext50_32x4d"], 'choice'),
            }

    best_hp_json_save_path: str
        Path to the json file where the best hyperparameters will be saved

    tuner_csv_save_path: str
        Path to the csv file where the hyperparameter tuning results will be saved.
        A modified version of the csv file will be saved in the same directory for sorted results

    tuner_obj_save_path: str
        Path to the file where the hyperparameter tuning object will be saved

    tune_target: str
        The metric to be optimised. This is the metric that will be used to find the best hyperparameters

    max_trials: int
        The maximum number of trials to be run for hyperparameter optimisation

    load_if_exists: bool
        Whether to load the tuner object from the tuner_obj_save_path if it exists or not.

    """

    global tune_cnt, total_tune_cnt, start_time

    if load_if_exists:
        print(f"Loading existing tuner object from {tuner_obj_save_path}")
    else:
        print(f"Creating new tuner object")

    tuner_utils = pt_tuner.HyperTunerUtils(
        best_hp_json_save_path=best_hp_json_save_path,
        tuner_csv_save_path=tuner_csv_save_path,
        tuner_obj_save_path=tuner_obj_save_path,
        tune_target=tune_target,
        tune_hp_ranges=search_space,
        max_trials=max_trials,
        train_function=train_to_tune,
        load_if_exists=load_if_exists,
        seed=0
    )

    tuner_utils.start_time = time.time()

    # Get the hp objects for each range in hyperopt
    search_space_hyperopt = tuner_utils.return_full_hp_dict(search_space)
    trials = Trials()

    best = fmin(
        tuner_utils.train_for_tuning,
        search_space_hyperopt,
        algo=tuner_utils.suggest_grid,
        max_evals=tuner_utils.max_trials,
        trials=trials,
        trials_save_file=tuner_utils.tuner_obj_save_path,
        verbose=True,
        show_progressbar=False
    )

    print("Best: ", best)
    print(space_eval(search_space_hyperopt, best))

    # Our pt_utils.hyper_tuner class will save the best hyperparameters to a json file after each trial


def visualise_generator(
        data_loader,
        full_labels=FULL_LABELS,
        num_images=None,
        model=None,
        model_save_path=MODEL_SAVE_PATH_BEST_VAL_LOSS,
        run_evaluation=True,
        val_batch_size=8,
        num_workers=MAX_THREADS,
        device=DEVICE,
        best_hp_json_save_path=BEST_HP_JSON_SAVE_PATH,
        num_classes=NUM_CLASSES,
):
    if type(data_loader) == str:
        if data_loader == 'train':
            data_generator = get_data_generators()[0]
        elif data_loader == 'val':
            data_generator = get_data_generators()[1]
    elif type(data_loader) != torch.utils.data.DataLoader:
        raise TypeError("data_loader must be of type str or torch.utils.data.DataLoader, or \"train\" or \"val\"")

    if type(data_loader) != torch.utils.data.DataLoader:
        data_loader = torch.utils.data.DataLoader(data_generator, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    if model is None:
        # load best hyperparameters
        best_hp = utils.load_dict_from_json(best_hp_json_save_path)

        model = models.models_dict[best_hp["conv_model"]](num_classes=num_classes, class_weights=None)
        model.load_state_dict(torch.load(model_save_path, map_location=device))

    model.eval()
    model.to(device)

    # evaluate model on data_loader
    if run_evaluation:
        print("\nEvaluating model on data_loader: ")
        results = pt_train._evaluate(model, data_loader)
        print("Results: ", results)

    cnt = 0
    for batch in data_loader:
        for (image, label) in zip(batch[0], batch[1]):

            print("\nImage: ", image.shape)
            print("Label: ", label)

            # get prediction
            image = image.unsqueeze(0).to(device)
            pred = model(image)
            print("NN output: ", pred)
            pred = int(torch.argmax(pred, dim=1).detach().cpu().numpy())
            print("Prediction: ", pred, "Pred name: ", full_labels[pred])

            image = image.squeeze(0).cpu().numpy()
            image = np.transpose(image, (1, 2, 0))

            label = int(torch.argmax(label, dim=0).detach().cpu().numpy())
            print("Label: ", label, "Label name: ", full_labels[label], "\n")

            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.title("Label name: " + str(full_labels[label]) + " | Pred name: " + str(full_labels[pred]))
            plt.show()

            cnt += 1
            if num_images and cnt >= num_images:
                return
