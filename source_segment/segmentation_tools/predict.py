import os
import shutil

import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt

# import source.segmentation_tools.tasm as tasm
import source.segmentation_tools.utils as seg_utils
import source.segmentation_tools.segmentation_config as seg_cf
import source.config as cf
import source.segmentation_tools.data_handling as data_handling


BATCH_SIZE = seg_cf.BATCH_SIZE
HEIGHT = seg_cf.HEIGHT
WIDTH = seg_cf.WIDTH
BACKBONE_NAME = seg_cf.BACKBONE_NAME
WEIGHTS = seg_cf.WEIGHTS
WWO_AUG = seg_cf.WWO_AUG  # train data with and without augmentation

MODEL_CLASSES = seg_cf.CHOSEN_MASKS.copy()
# print("MODEL_CLASSES: ", MODEL_CLASSES)
N_CLASSES = len(seg_cf.CHOSEN_MASKS)


def predict_image(image, model, display_image=False, prediction_th=seg_cf.PREDICTION_TH, mode_360=False):
    image_org = image.copy()
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

    input("enter")
    plt.imshow(image)
    plt.show()

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = torch.tensor(image, dtype=torch.float32).to(seg_cf.DEVICE)
    print("image shape", image.shape)

    output_model = model(image)

    output_model = output_model[0]
    output_model = output_model.cpu().detach().numpy()
    output_model = np.transpose(output_model, (1, 2, 0))

    print("output model shape", output_model.shape)

    display_list, title_list, masks_list = [], [], []
    display_list.append(image)
    title_list.append("Image")

    # output_model[output_model > prediction_th] = 1
    # output_model[output_model <= prediction_th] = 0

    arg_max_arrays = np.argmax(output_model[..., :], axis=-1)
    # arg_max_arrays = [arg_max_arrays_1, arg_max_arrays_2]

    super_imposed = image_org * 255.
    for i in range(1, output_model.shape[-1]):
        arg_max_array = arg_max_arrays
        layer_of_interest = output_model[..., i].copy()

        if not mode_360:
            layer_of_interest[arg_max_array != i] = 0
            layer_of_interest[arg_max_array == i] = 1

        print("layer_of_interest shape", layer_of_interest.shape, np.sort(np.unique(layer_of_interest)))
        colours = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)
        colour = colours[i]

        # super_imposed = seg_utils.superimpose_mask_on_image(super_imposed, layer_of_interest, colour=colour).copy()
        super_imposed = seg_utils.super_impose_RGBA(super_imposed, (1 - layer_of_interest) * 255.)

        layer_of_interest = layer_of_interest[..., tf.newaxis]
        title_list.append(f"Mask {i}")
        display_list.append(layer_of_interest)
        masks_list.append(layer_of_interest)

    title_list.append("Superimposed")
    display_list.append(super_imposed)
    if display_image:
        seg_utils.display(display_list, title_list)

    return super_imposed, masks_list


def run_prediction_on_generator(model, pred_set, display_image=False, prediction_th=seg_cf.PREDICTION_TH, mode_360=False):
    super_imposed_list = []
    masks_list = []
    file_name_list = []
    for i in pred_set:
        if i is None:
            break

        image, path = i
        print("img_shape", image.shape)
        super_imposed, masks = predict_image(image, model, display_image=display_image, prediction_th=prediction_th, mode_360=mode_360)
        print("super_imposed shape", super_imposed.shape)
        super_imposed_list.append(super_imposed)
        masks_list.append(masks)
        file_name_list.append(path.split(os.sep)[-1])

    print("Reached end of generator")
    return super_imposed_list, masks_list, file_name_list


def run_images(
        input_images_folder=cf.FLOOR_DETECT_INPUT_FOLDER, prediction_dir=cf.FLOOR_DETECT_OUTPUT_FOLDER, display_image=False, save_name_ext=".png", mode_360=False, prediction_th=seg_cf.PREDICTION_TH
):
    prediction_generator = data_handling.SimpleGenerator(input_images_folder)

    # base_model, layers, layer_names = tasm.tf_backbones.create_base_model(name=BACKBONE_NAME, weights=WEIGHTS, height=HEIGHT, width=WIDTH, include_top=False, pooling=None)
    # model = tasm.DeepLabV3plus.DeepLabV3plus(n_classes=N_CLASSES, base_model=base_model, output_layers=layers, backbone_trainable=True)

    # model = deepvision.models.SegFormerB1(
    #     input_shape=(HEIGHT, WIDTH, 3),
    #     num_classes=N_CLASSES,
    #     softmax_output=True,
    #     backend='tensorflow'
    # )

    # categorical_focal_dice_loss = tasm.losses.CategoricalFocalLoss(alpha=0.25, gamma=2.0) + tasm.losses.DiceLoss()
    # model.run_eagerly = True
    # model.load_weights(cf.MODEL_SAVE_PATH).expect_partial()

    model = torch.load(cf.MODEL_SAVE_PATH).to(seg_cf.DEVICE)

    if os.path.exists(prediction_dir):
        shutil.rmtree(prediction_dir, ignore_errors=True)
    os.makedirs(prediction_dir)

    if os.path.exists(prediction_dir + f"comb_mask"):
        shutil.rmtree(prediction_dir + f"comb_mask", ignore_errors=True)
    os.makedirs(prediction_dir + f"comb_mask")

    super_imposed_list, masks_list, file_name_list = run_prediction_on_generator(model, prediction_generator, display_image=display_image, prediction_th=prediction_th, mode_360=mode_360)

    for i, super_imposed in enumerate(super_imposed_list):
        file_name_ext = "." + file_name_list[i].split(".")[-1]
        file_name_list[i] = file_name_list[i].replace(file_name_ext, "")

        Image.fromarray(super_imposed.astype(np.uint8)).save(prediction_dir + f"superimp_{file_name_list[i]}{save_name_ext}")

        empty_mask = np.zeros(masks_list[i][0].shape[:2])
        for j, mask in enumerate(masks_list[i]):
            mask = mask[..., 0]
            if not os.path.exists(prediction_dir + f"mask_{j}"):
                os.makedirs(prediction_dir + f"mask_{j}")
            cv2.imwrite(prediction_dir + f"mask_{j}{os.sep}" + f"pred_{file_name_list[i]}_mask_{j}{save_name_ext}", mask * 255)
            empty_mask += mask

        if mode_360:
            # we are marking it as one so that we can differentiate the non floor pixels from the pixels not in the FOV of the predicted area of the 360 image
            rgb_mask = np.zeros((empty_mask.shape[0], empty_mask.shape[1], 3))
            # copy the mask that has a range of 0 to 255 to all the channels
            empty_mask *= 255
            rgb_mask[..., 0] = empty_mask
            rgb_mask[..., 1] = empty_mask
            rgb_mask[..., 2] = empty_mask
        else:
            empty_mask[empty_mask > 0] = 255
            rgb_mask = np.zeros((empty_mask.shape[0], empty_mask.shape[1], 3))
            rgb_mask[empty_mask == 255] = (255, 255, 255)

        comb_mask_file_path = prediction_dir + f"comb_mask{os.sep}" + f"pred_{file_name_list[i]}.CombMask{i}{save_name_ext}"
        print("empty_mask shape", rgb_mask.shape, np.sort(np.unique(rgb_mask)), comb_mask_file_path)
        cv2.imwrite(comb_mask_file_path, rgb_mask)


if __name__ == '__main__':
    # data_handling.init()
    run_images()
