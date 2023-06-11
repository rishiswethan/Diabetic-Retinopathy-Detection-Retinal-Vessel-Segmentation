
import numpy as np
import torch
from torch.utils.data import DataLoader


import source_segment.segmentation_tools.segmentation_models_pytorch as smp
import source_segment.segmentation_tools.segmentation_models_pytorch.utils as smp_utils
import source_segment.segmentation_tools.segmentation_models_pytorch.losses as smp_losses

import source_segment.config as cf
# import source_segment.segmentation_tools_pytorch.tasm as tasm
import source_segment.segmentation_tools.utils as seg_utils
import source_segment.segmentation_tools.data_handling as data_handling
import source_segment.segmentation_tools.segmentation_config as seg_cf

import source_segment.segmentation_tools.pytorch_utils.training_utils as pt_train
import source_segment.segmentation_tools.pytorch_utils.callbacks as pt_callbacks

# find the device to be used for training from tensorflow and set memory growth to true
has_gpu = torch.cuda.is_available()
device_torch = torch.device("cuda" if has_gpu else "cpu")
# num_cpu = multiprocessing.cpu_count() // 4
num_cpu = 2
# print("Number of CPUs: ", num_cpu)

# define initial variables
MODEL_CLASSES = seg_cf.CHOSEN_MASKS.copy()

N_CLASSES = len(seg_cf.CHOSEN_MASKS)

BATCH_SIZE = seg_cf.BATCH_SIZE
HEIGHT = seg_cf.HEIGHT
WIDTH = seg_cf.WIDTH
BACKBONE_NAME = seg_cf.BACKBONE_NAME
WEIGHTS = seg_cf.WEIGHTS
WWO_AUG = seg_cf.WWO_AUG  # train data with and without augmentation
PROB_APPLY_AUGMENTATION = seg_cf.PROB_APPLY_AUGMENTATION


def get_data_generators(batch_size, height, width, classes=MODEL_CLASSES, train_shuffle=True, val_shuffle=True, seed=None):
    TrainSet = data_handling.DataGenerator(
        'train',
        batch_size,
        height,
        width,
        classes=classes,
        augmentation=data_handling.get_training_augmentation(height=height, width=width),
        prob_apply_aug=PROB_APPLY_AUGMENTATION,
        shuffle=train_shuffle,
        seed=seed,
        verbose=False
    )

    ValidationSet = data_handling.DataGenerator(
        'test',
        batch_size,
        height,
        width,
        classes=classes,
        augmentation=data_handling.get_validation_augmentation(height=height, width=width),
        shuffle=val_shuffle,
        seed=seed,
        verbose=False
    )

    return TrainSet, ValidationSet


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
                                          model_save_path=cf.MODEL_SAVE_PATH_BEST_VAL_LOSS,
                                          training_stats_path=cf.VAL_CALLBACK_OBJ_PATH,
                                          continue_training=continue_training),
            'train': pt_callbacks.Callbacks(optimizer=optimiser,
                                            model_save_path=cf.MODEL_SAVE_PATH_BEST_TRAIN_LOSS,
                                            training_stats_path=cf.TRAIN_CALLBACK_OBJ_PATH,
                                            continue_training=continue_training)
        }

    defined_callbacks['val'].reduce_lr_on_plateau(
        monitor_value=result["val_acc"],
        mode='max',
        factor=seg_cf.REDUCE_LR_FACTOR_VAL,
        patience=seg_cf.REDUCE_LR_PATIENCE_VAL,
        indicator_text="Val LR scheduler: "
    )
    defined_callbacks['train'].reduce_lr_on_plateau(
        monitor_value=result["train_acc"],
        mode='max',
        factor=seg_cf.REDUCE_LR_FACTOR_TRAIN,
        patience=seg_cf.REDUCE_LR_PATIENCE_TRAIN,
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
        monitor_value=result[seg_cf.EARLY_STOPPING_MONITOR],
        mode='min',
        patience=seg_cf.EARLY_STOPPING_PATIENCE,
        indicator_text="Early stopping: "
    )
    defined_callbacks['train'].clear_memory()
    print("_________")

    return defined_callbacks, stop_flag


def predict_image(image_np, model):
    device = next(model.parameters()).device  # Get the device the model is on
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float().to(device)  # Move the input tensor to the same device as the model
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)

    return output.cpu()


def _train(
        continue_training=False,
        load_weights_for_fine_tune=False,
        activation=seg_cf.ACTIVATION,
        device=seg_cf.DEVICE,
        INITIAL_LR=seg_cf.INITIAL_LR,
        train_model_save_path=cf.MODEL_SAVE_PATH_BEST_TRAIN_LOSS,
        val_model_save_path=cf.MODEL_SAVE_PATH_BEST_VAL_LOSS,
        input_shape=(3, HEIGHT, WIDTH),
):
    """
    Train the model

    :param continue_training: bool, if True, continue training from the last saved model. All training stats, including checkpoint stats will be loaded
    :param load_weights_for_fine_tune: bool, if True, load weights from the last saved model. All training stats will be reset
    """

    TrainSet, ValidationSet = get_data_generators(BATCH_SIZE, HEIGHT, WIDTH, classes=MODEL_CLASSES, train_shuffle=True, val_shuffle=True, seed=None)

    # choose train set with or without augmentation or one with and one without augmentation
    chosen_train_set = TrainSet

    chosen_train_set = DataLoader(chosen_train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_cpu)
    ValidationSet = DataLoader(ValidationSet, batch_size=BATCH_SIZE, shuffle=False)

    print("TrainSet length: ", len(chosen_train_set))
    print("ValidationSet length: ", len(ValidationSet))

    class_weights = seg_utils.get_balancing_class_weights(seg_cf.CHOSEN_MASKS, data_handling.CLASSES_PIXEL_COUNT_DICT)
    class_weights = class_weights[:N_CLASSES]

    # class_weights = [1, 1]  # Can be disabled by setting all weights to 1

    print("class_weights", class_weights)

    if continue_training or load_weights_for_fine_tune:
        model = torch.load(train_model_save_path)
    else:
        model = smp.DeepLabV3Plus(
            encoder_name=seg_cf.BACKBONE_NAME,
            encoder_weights=seg_cf.WEIGHTS,
            classes=len(seg_cf.CHOSEN_MASKS),
            activation=seg_cf.ACTIVATION,
        )

    # visualize the model
    model.eval()
    for i, data_point in enumerate(chosen_train_set):
        for j in range(len(data_point[0])):
            print("i", i, data_point[0].shape, data_point[1].shape)
            image_, mask = data_point[0][j], data_point[1][j]
            print("image", image_.shape)
            print("mask", mask.shape)

            image = np.array(image_)
            mask = np.array(mask)

            # unsqueeze the single image and mask with (3, 256, 256) to (256, 256, 3)
            image = np.transpose(image, (1, 2, 0))
            mask = np.transpose(mask, (1, 2, 0))

            # convert the one hot encoded mask to a single channel mask
            mask = np.argmax(mask, axis=2)

            print("image", image.shape, np.unique(image))
            print(mask.shape)

            output = predict_image(image, model)
            output = np.transpose(output[0], (1, 2, 0))
            print("output unique: ", np.unique(output))
            output = np.argmax(output, axis=2)
            print(output.shape)

            seg_utils.visualize(image=image, mask=mask, output=output)

            break
        break

    loss = smp.utils.losses.DiceLoss()

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0]),
    ]

    if continue_training:
        objects = pt_callbacks.load_saved_objects(cf.VAL_CALLBACK_OBJ_PATH)
        optimizer = objects.optimizer
        other_stats = objects.other_stats
        initial_epoch = other_stats['epochs'] + 1  # +1 because the epoch is incremented at the end of the loop
    else:
        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=seg_cf.INITIAL_LR),
        ])
        initial_epoch = 0

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    defined_callbacks = None
    for i in range(initial_epoch, 99999999):
        print(f'\nEpoch: {i + 1} LR: {optimizer.param_groups[0]["lr"]}\n')
        train_logs = train_epoch.run(chosen_train_set)
        valid_logs = valid_epoch.run(ValidationSet)

        results = {
            'train_acc': train_logs['iou_score'],
            'val_acc': valid_logs['iou_score'],
            'train_loss': train_logs['dice_loss'],
            'val_loss': valid_logs['dice_loss'],
        }

        other_stats = {"epochs": i}

        defined_callbacks, stop_flag = get_callbacks(
            optimiser=optimizer,
            result=results,
            model=model,
            defined_callbacks=defined_callbacks,
            continue_training=continue_training,
            other_stats=other_stats,
        )
        if stop_flag:
            print("Early stopping triggered")
            break


def evaluate_model(model, eval_set, metrics, class_weights=None):
    for i in eval_set:
        sample_image, sample_mask = i[0][0], i[1][0]
        print(len(i))
        print(i[0].shape)
        print(i[1].shape)

        model.evaluate(x=i[0], y=i[1], verbose=1, steps=len(i))

        data_handling.show_sample_predictions(
            sample_image=sample_image,
            sample_mask=sample_mask,
            model=model,
            class_weights=class_weights,
        )


# ReduceLROnPlateau saves the last used lr and model uses it by default. This lets us start from the initial lr
def fix_scheduler_initial_lr(epoch, lr):
    if epoch == 0 and lr != seg_cf.INITIAL_LR:
        return seg_cf.INITIAL_LR
    else:
        return lr


def train(
        continue_training,
        load_weights_for_fine_tune,
):
    data_handling.init()
    _train(continue_training=continue_training, load_weights_for_fine_tune=load_weights_for_fine_tune)


if __name__ == "__main__":
    if has_gpu:
        print("\n======> GPU is available and in use <========\n")
    else:
        print("\n======> GPU is not available, program will still run on the CPU <=====\n")
    print("Num CPUs Available: ", num_cpu)

    data_handling.init()
    train(continue_training=True, load_weights_for_fine_tune=False)

    # Test the model on the test set and show some predictions. This is used for debugging and testing purposes and will be removed in the final version
    # ValidationSet = data_handling.DataGenerator(
    #     'test',
    #     1,
    #     HEIGHT,
    #     WIDTH,
    #     classes=MODEL_CLASSES,
    #     augmentation=data_handling.get_validation_augmentation(height=HEIGHT, width=WIDTH),
    #     shuffle=True,
    #     verbose=True  # Make verbose true to see images and masks
    # )
    #
    # for i in ValidationSet:
    #     print(len(i))
    #     print(i[0].shape)
    #     print(i[1].shape)

    # base_model, layers, layer_names = tasm.tf_backbones.create_base_model(name=BACKBONE_NAME, weights=WEIGHTS, height=HEIGHT, width=WIDTH, include_top=False, pooling=None)
    # class_weights = seg_utils.get_balancing_class_weights(MODEL_CLASSES, data_handling.CLASSES_PIXEL_COUNT_DICT)
    # # class_weights[2] = 0.00001
    # print(class_weights)
    # model = tasm.DeepLabV3plus.DeepLabV3plus(n_classes=N_CLASSES, base_model=base_model, output_layers=layers, backbone_trainable=True)
    #
    # opt = tf.keras.optimizers.Adam(learning_rate=seg_cf.INITIAL_LR)
    # metrics = [tasm.metrics.IOUScore(threshold=0.5, class_weights=class_weights), "accuracy"]
    # categorical_focal_dice_loss = tasm.losses.CategoricalFocalLoss(alpha=0.25, gamma=2.0) + tasm.losses.DiceLoss()
    # model.compile(
    #     # optimizer=opt,
    #     loss=categorical_focal_dice_loss,
    #     metrics=metrics,
    #     run_eagerly=True
    # )
    # model.run_eagerly = True
    # model.load_weights(cf.MODEL_SAVE_PATH)
    # # model.evaluate(ValidationSet, verbose=1, steps=len(ValidationSet), sample_weight=class_weights)
    # evaluate_model(model, ValidationSet, metrics=metrics, class_weights=class_weights)
