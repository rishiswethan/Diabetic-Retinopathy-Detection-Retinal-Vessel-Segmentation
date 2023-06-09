#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import cv2
import shutil

import source.tools_360.convert as convert
import source.config as config
import source.segmentation_tools.segmentation_config as seg_cf
from source.tools_360 import detect_borders as db
import source.tools_360.multi_p2e as m_P2E
import source.utils as utils

import source.segmentation_tools.predict as ai_predict

equi_letters = ['+Z', '+X', '-Z', '-X', '+Y', '-Y']  # 0F 1R 2B 3L 4U 5D


# Converts the equirectangular image to cubemap
def equi_cubemap(img_name=os.path.join(config.INPUT_FOLDER, "eq_image.png"),
                 output_image_folder=config.E2C_OUTPUT_FOLDER,
                 output_image_prefix=config.E2C_OUTPUT_IMAGE_PREFIX):
    img_path = img_name
    img = np.array(Image.open(img_path))
    square_resolution = round(min(img.shape[:2]))

    out = convert.equi_cubemap(img, face_w=square_resolution, mode="bilinear", cube_format="list")
    print(f"Output length: {len(out)}")
    for i, image in enumerate(out):
        # Output image
        output_image_path = output_image_folder + output_image_prefix + f"_{equi_letters[i]}.png"
        Image.fromarray(np.array(image)).save(output_image_path)


# Converts the multiple cubemap images to equirectangular
def cubemap_equi(input_folder=config.INPUT_FOLDER, input_image_prefix=config.E2C_OUTPUT_IMAGE_PREFIX, output_image_folder=config.C2E_OUTPUT_FOLDER):
    suffix_list = [f"_{equi_letters[i]}" for i in range(6)]

    image_list = []
    for i, suffix in enumerate(suffix_list):
        image_name = utils.find_filename_match(input_image_prefix + suffix, input_folder)
        image = np.array(Image.open(image_name).convert('RGB'))
        print(suffix, image.shape)

        image_list.append(image)

    h = image.shape[0]
    w = image.shape[0] * 2

    # make h and w divisible by 8
    h = h - h % 8
    w = w - w % 8

    out = convert.cubemap_equi(image_list, h=h, w=w, mode='bilinear', cube_format="list")

    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    output_image_name = output_image_folder + f"c2e.png"
    Image.fromarray(out.astype(np.uint8)).save(output_image_name)


# Converts the equirectangular image to a zoomed in place according to the config file in config.PERSPECTIVE_CONFIG_JSON_PATH
def equi_perspective(img_path, config_dict=utils.load_dict_from_json(config.PERSPECTIVE_CONFIG_JSON_PATH), output_path=config.E2P_OUTPUT_IMAGE_PATH):
    img = np.array(Image.open(img_path))
    img = img[:, :, :3]  # remove alpha channel
    print(img.shape)
    out = convert.equi_perspective(img,
                                   field_of_view=(config_dict['h_field_of_view'], config_dict['v_field_of_view']),  # Field of view given in int or tuple (horizontal, vertical)
                                   horizontal_view_angle=config_dict['horizontal_viewing_angle'],  # Horizontal viewing angle
                                   vertical_view_angle=config_dict['vertical_viewing_angle'],  # Vertical viewing angle
                                   out_hw=config_dict['output_image_size'],  # Output image size
                                   in_rot_deg=config_dict['image_rotate'],  # Rotation of input image in degrees
                                   mode='bilinear')

    save_name_ext = "." + output_path.split(".")[-1]
    save_name = output_path.replace(save_name_ext,
                                    f"_h{config_dict['horizontal_viewing_angle']}_v{config_dict['vertical_viewing_angle']}_hfov{config_dict['h_field_of_view']}_vfov{config_dict['v_field_of_view']}{save_name_ext}")

    # delete the old image
    if os.path.exists(save_name):
        os.remove(save_name)

    Image.fromarray(out.astype(np.uint8)).save(save_name)


# Combines multiple perspective images into one equirectangular image
def pers_equi(input_dir=config.P2E_INPUT_FOLDER, output_dir=config.P2E_OUTPUT_FOLDER, opt_width=5000, opt_height=2500, skip_roof=True, additional_skip_range=None):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    empty_image = None

    input_images = []
    img_attributes = []
    mask_images = []
    for file in os.listdir(input_dir):
        if os.path.isdir(input_dir + file):
            continue

        if empty_image is None:
            img = cv2.imread(input_dir + file, cv2.IMREAD_UNCHANGED)
            empty_image = np.zeros(img.shape, dtype=np.uint8)
            empty_image.fill(0)
            # save the empty image
            cv2.imwrite(output_dir + "empty_image.png", empty_image)

            white_image = np.zeros(img.shape, dtype=np.uint8)
            white_image.fill(255)
            # save the white image
            cv2.imwrite(output_dir + "white_image.png", white_image)

        print(file)

        v = float(file.split('_')[-3][1:].split('.')[0])
        h = float(file.split('_')[-4][1:])
        vfov = int(file.split('_')[-1].split('.')[0][4:])
        hfov = int(file.split('_')[-2][4:])

        if skip_roof and 100 > v >= 0:
            print(f"Skipping image: v={v}, h={h}, hfov={hfov}, vfov={vfov}")
            # append an empty image path to keep the order
            input_images.append(output_dir + "empty_image.png")
            img_attributes.append([hfov, h, v])

        elif additional_skip_range is not None and additional_skip_range[0] <= v <= additional_skip_range[1]:
            print(f"Skipping image: v={v}, h={h}, hfov={hfov}, vfov={vfov}")
            # append an empty image path to keep the order
            input_images.append(output_dir + "white_image.png")
            img_attributes.append([hfov, h, v])

        else:
            img_attributes.append([hfov, h, v])
            input_images.append(input_dir + file)
            # Load image and convert to numpy array
            img = cv2.imread(input_dir + file)
            print(img.shape)

    equ = m_P2E.Perspective(input_images,
                            img_attributes)

    print("\nConverting to equirectangular...")
    img = equ.get_equirec(opt_height, opt_width)
    print(img.shape, img.dtype, np.unique(img))
    # delete the old image
    if os.path.exists(output_dir + "p2e_output.png"):
        os.remove(output_dir + "p2e_output.png")

    cv2.imwrite(output_dir + 'p2e_output.png', img)
    # delete the empty image
    if os.path.exists(output_dir + "empty_image.png"):
        os.remove(output_dir + "empty_image.png")


def correct_360_mask_and_combine(mask_path, actual_360_image, combined_360_opt_path, overlap_th=50, skip_primary_mask=None, primary_pred_fov=90, blur_factor=seg_cf.BLUR_FACTOR):
    # num_overlap_th is the % pixel overlap required to be considered as an overlap

    img = np.array(Image.open(mask_path), dtype=np.float32)
    print("Correcting 360 mask")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.shape, np.unique(img))

    img_cpy = np.zeros_like(img)

    # masks for floor are white(255). In RGBA, white indicates visible, so we need to make it black (i.e. 0)
    # Higher the overlap_th, more masks need to overlap to be considered as an overlap, since it requires more masks to make it this "bright"
    print("overlap_th", overlap_th)

    img_cpy = 255 - img
    range_start = overlap_th
    range_end = 255

    blur = (((img_cpy - overlap_th) / (range_end - range_start)) * blur_factor)
    less_than_overlap_th = img_cpy <= overlap_th
    more_than_overlap_th = img_cpy > overlap_th + (0.5 * (255 - overlap_th))

    # make anything over the overlap_th value to be the blur value in the range of 0 to half of the blur_factor
    img_cpy = np.where(img_cpy > overlap_th, blur, img_cpy)

    img_cpy[less_than_overlap_th] = 0
    img_cpy[more_than_overlap_th] = 255

    border_index = None
    if skip_primary_mask is None:
        mask_covered_fov_indices = np.zeros_like(img, dtype=np.uint8)
        mask_covered_fov_indices[img != 0] = 1

        max_border_index = 0
        for i in range(0, img_cpy.shape[1]):
            if len(np.where(mask_covered_fov_indices[:, i] == 1)[0]) == 0:
                continue

            border_index = np.where(mask_covered_fov_indices[:, i] == 1)[0][-1]

            if border_index > max_border_index:
                max_border_index = border_index

        border_index = max_border_index
        print("border_index", border_index)
        print("mask_covered_fov_indices.shape", mask_covered_fov_indices.shape, np.unique(mask_covered_fov_indices))

        # mask_covered_fov_indices = np.where(mask_covered_fov_indices != 0, 1, 0).astype(np.uint8)
        border_index -= 10

        # plt.imshow(mask_covered_fov_indices)
        # plt.show()

        print("border_index", border_index)
        # plt.imshow(mask_covered_fov_indices)
        # plt.show()

    image_360 = np.array(Image.open(actual_360_image))
    image_360 = image_360[:, :, :3]
    print("image_360.shape", image_360.shape)

    # add a new channel to the image
    image_360 = np.concatenate((image_360, np.zeros((image_360.shape[0], image_360.shape[1], 1), dtype=np.uint8)), axis=2)
    print("image_360.shape", image_360.shape)

    if skip_primary_mask is not None:
        print("Making this range white mask:", skip_primary_mask.shape, skip_primary_mask)

        img_cpy[:skip_primary_mask, :] = 255
        skip_primary_mask = skip_primary_mask / img_cpy.shape[0]
        border_index = int(skip_primary_mask * image_360.shape[0]) + 1

    print("img_cpy.shape", img_cpy.shape, np.unique(img_cpy))
    img_cpy = cv2.resize(img_cpy, (image_360.shape[1], image_360.shape[0]), interpolation=cv2.INTER_NEAREST)
    image_360[:, :, 3] = img_cpy
    print("image_360.shape", image_360.shape, np.unique(image_360[:, :, 3]))
    # plt.imshow(image_360)
    # plt.show()

    img = Image.fromarray(image_360)
    img.save(combined_360_opt_path)

    return border_index


def crop_360_directly(input_path, buffer_folder=config.BUFFER_FOLDER):
    # clear buffer folder
    if os.path.exists(buffer_folder):
        shutil.rmtree(buffer_folder)
    os.mkdir(buffer_folder)

    img_ = np.array(Image.open(input_path))
    img = img_[..., :3]
    print("Cropping 360 image")
    print(img.shape)

    # crop the rectangle into 2 squares
    images = []
    for i in range(2):
        images.append(img[:,
                          img.shape[1] // 2 * i: img.shape[1] // 2 * (i + 1)])
        print(images[-1].shape, i)
        Image.fromarray(images[-1]).save(buffer_folder + f"image_{i}.jpg")

    return buffer_folder


def combine_cropped_360(mask_input_folder, image_input_path, output_image_prefix, output_folder):
    left = cv2.imread(mask_input_folder + "pred_image_0.CombMask0.png", cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(mask_input_folder + "pred_image_1.CombMask1.png", cv2.IMREAD_GRAYSCALE)

    out = np.concatenate((left, right), axis=1)

    out = 255.0 - out
    img = np.array(Image.open(image_input_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    out = cv2.resize(out, (img.shape[1], img.shape[0]))

    # add a new channel to the image
    img = np.concatenate((img, np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)), axis=2)
    print("image_360.shape", img.shape)
    print("out.shape", out.shape)
    img[:, :, 3] = out

    cv2.imwrite(output_folder + output_image_prefix + ".png", img)


# Detects the borders of the image, saves the pixels in a csv image with the borders highlighted
def detect_border(img_path):
    img = np.array(Image.open(img_path))
    print("Detecting borders")
    db.get_border(img)


def extract_multiple_perspectives(
        input_path,
        fov,
        max_pred_fov,
):
    """
    Extracts multiple perspectives from a 360 image
    """
    additional_skip_range = None

    h_field_of_view = fov
    v_field_of_view = fov

    # Extract perspective images for the main prediction fov. Typically, 90 degrees
    print("From 360 to 270 degrees")

    output_path = config.E2P_OUTPUT_FOLDER + "1" + os.sep + "e2p.png"
    utils.delete_folder_contents(config.E2P_OUTPUT_FOLDER + "1" + os.sep)

    def_centre_v = seg_cf.INFERENCE_360_V_VIEWING_ANGLE_MAIN
    h_increment = seg_cf.INFERENCE_360_PERSPECTIVE_MAIN_H_INC
    for i in np.arange(0, 360, h_increment):
        print(
            f"Extracting image with centre_h: {i}, centre_v: {def_centre_v}, h_fov: {h_field_of_view}, v_fov: {v_field_of_view}")
        equi_perspective(input_path,
                         {
                             "horizontal_viewing_angle": i,
                             "vertical_viewing_angle": def_centre_v,
                             "h_field_of_view": h_field_of_view,
                             "v_field_of_view": v_field_of_view,
                             "output_image_size": [seg_cf.PERSPECTIVE_SQUARE_SIZE, seg_cf.PERSPECTIVE_SQUARE_SIZE],
                             "image_rotate": 0
                         },
                         output_path=output_path)

    # Extract perspective images for the max prediction fov. Typically, 140 degrees. This will most of the fov missed by the main prediction fov.
    output_path = config.E2P_OUTPUT_FOLDER + "2" + os.sep + "e2p.png"
    utils.delete_folder_contents(config.E2P_OUTPUT_FOLDER + "2" + os.sep)

    def_centre_v = seg_cf.INFERENCE_360_V_VIEWING_ANGLE_MAX
    v_field_of_view = h_field_of_view = seg_cf.INFERENCE_360_PERSPECTIVE_FOV_MAX

    h_increment = seg_cf.INFERENCE_360_PERSPECTIVE_MAX_H_INC
    for i in np.arange(0, 360, h_increment):
        print(
            f"Extracting image with centre_h: {i}, centre_v: {def_centre_v}, h_fov: {h_field_of_view}, v_fov: {v_field_of_view}")
        equi_perspective(input_path,
                         {
                             "horizontal_viewing_angle": i,
                             "vertical_viewing_angle": def_centre_v,
                             "h_field_of_view": h_field_of_view,
                             "v_field_of_view": v_field_of_view,
                             "output_image_size": [seg_cf.PERSPECTIVE_SQUARE_SIZE, seg_cf.PERSPECTIVE_SQUARE_SIZE],
                             "image_rotate": 0
                         },
                         output_path=output_path)

    lower_skp_fov = ((def_centre_v - (v_field_of_view / 2)) - 270) * 2
    lower_skp_fov += 2  # Add a small buffer
    lower_skp_fov = int(lower_skp_fov)
    skip_centre_v = 270

    if lower_skp_fov > 5:
        # This is to cover the remaining fov, usually only a small range, like 20 degrees. This will be marked as "skip" or floor.
        print(f"Extracting image with centre_h: {i}, centre_v: {skip_centre_v}, h_fov: {lower_skp_fov}, v_fov: {lower_skp_fov}")
        equi_perspective(input_path,
                            {
                                "horizontal_viewing_angle": i,
                                "vertical_viewing_angle": skip_centre_v,
                                "h_field_of_view": lower_skp_fov,
                                "v_field_of_view": lower_skp_fov,
                                "output_image_size": [128, 128],
                                "image_rotate": 0
                            },
                            output_path=output_path)

    additional_skip_range = [270, 270 + lower_skp_fov]  # This is the range of fov that will be marked as "skip" or floor.

    return additional_skip_range


def combine_2_equi_rgba_masks(img1_path, img2_path, output_path, skip_roof=True, combine_border_index=None):
    img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)

    img1[:, :, 3] = 255 - img1[:, :, 3]
    img2[:, :, 3] = 255 - img2[:, :, 3]

    print("combine_border_index: ", combine_border_index)

    combined_rgba = img1.copy()
    combined_rgba[:, :, 3][:combine_border_index] = img1[:, :, 3][:combine_border_index]
    combined_rgba[:, :, 3][combine_border_index:] = img2[:, :, 3][combine_border_index:]

    combined_rgba[:, :, 3] = 255 - combined_rgba[:, :, 3]

    cv2.imwrite(output_path, combined_rgba)


# Extracts multiple perspective images from the equirectangular image, runs the UNet segmentation model on the images and stitches them back together
def extract_and_predict(input_path,
                        output_path=config.C2E_OUTPUT_FOLDER,
                        fov=seg_cf.INFERENCE_360_PERSPECTIVE_FOV_MAIN,
                        max_fov=seg_cf.INFERENCE_360_PERSPECTIVE_FOV_MAX,
                        predict=False,
                        skip_roof=seg_cf.INFERENCE_360_SKIP_ROOF,
                        extract_mode=seg_cf.EXTRACTION_MODE_360,
                        copy_to_simple_output=True,):
    # delete files in output folder
    for file in os.listdir(config.E2P_OUTPUT_FOLDER):
        if os.path.isdir(config.E2P_OUTPUT_FOLDER + file):
            continue
        os.remove(config.E2P_OUTPUT_FOLDER + file)

    for file in os.listdir(config.FLOOR_DETECT_OUTPUT_FOLDER + "360" + os.sep):
        if os.path.isdir(config.FLOOR_DETECT_OUTPUT_FOLDER + "360" + os.sep + file):
            continue
        os.remove(config.FLOOR_DETECT_OUTPUT_FOLDER + "360" + os.sep + file)

    if extract_mode == "perspective":
        output_path = config.P2E_OUTPUT_FOLDER
        additional_skip_range = extract_multiple_perspectives(input_path, fov=fov, max_pred_fov=max_fov)

    elif extract_mode == "cubemap":
        output_path = config.C2E_OUTPUT_FOLDER
        equi_cubemap(input_path,
                     output_image_folder=config.E2C_OUTPUT_FOLDER,
                     output_image_prefix=config.E2C_OUTPUT_IMAGE_PREFIX)

    elif extract_mode == "simple_crop":
        output_path = config.BUFFER_FOLDER
        crop_360_directly(input_path)

    if predict:
        if not os.path.exists(config.FLOOR_DETECT_OUTPUT_FOLDER + "360" + os.sep):
            os.mkdir(config.FLOOR_DETECT_OUTPUT_FOLDER + "360" + os.sep)

        if extract_mode == "perspective":
            # detect floor using segmentation model
            ai_predict.run_images(input_images_folder=config.E2P_OUTPUT_FOLDER + "1" + os.sep,
                                  prediction_dir=config.FLOOR_DETECT_OUTPUT_FOLDER + "360" + os.sep + "1" + os.sep,
                                  mode_360=True)
            pers_equi(
                input_dir=config.FLOOR_DETECT_OUTPUT_FOLDER + "360" + os.sep + "1" + os.sep + "comb_mask" + os.sep,
                output_dir=output_path + "comb_mask" + os.sep,
                skip_roof=skip_roof,
            )
            primary_mask = correct_360_mask_and_combine(
                mask_path=output_path + "comb_mask" + os.sep + "p2e_output.png",
                actual_360_image=input_path,
                combined_360_opt_path=output_path + "p2e_output_1.png",
                overlap_th=seg_cf.INFERENCE_MAIN_OVERLAP_TH,
                primary_pred_fov=fov
            )

            ai_predict.run_images(input_images_folder=config.E2P_OUTPUT_FOLDER + "2" + os.sep,
                                  prediction_dir=config.FLOOR_DETECT_OUTPUT_FOLDER + "360" + os.sep + "2" + os.sep,
                                  mode_360=True,
                                  prediction_th=seg_cf.PREDICTION_TH_MAX_FOV)
            pers_equi(
                input_dir=config.FLOOR_DETECT_OUTPUT_FOLDER + "360" + os.sep + "2" + os.sep + "comb_mask" + os.sep,
                output_dir=output_path + "comb_mask" + os.sep,
                skip_roof=skip_roof,
                additional_skip_range=additional_skip_range
            )
            border_index = correct_360_mask_and_combine(
                mask_path=output_path + "comb_mask" + os.sep + "p2e_output.png",
                actual_360_image=input_path,
                combined_360_opt_path=output_path + "p2e_output_2.png",
                overlap_th=seg_cf.INFERENCE_MAX_OVERLAP_TH,
                skip_primary_mask=primary_mask,
                primary_pred_fov=fov,
            )

            combine_2_equi_rgba_masks(
                img1_path=output_path + "p2e_output_1.png",
                img2_path=output_path + "p2e_output_2.png",
                output_path=output_path + "p2e_output.png",
                skip_roof=skip_roof,
                combine_border_index=border_index
            )
            if copy_to_simple_output:
                utils.copy_to_simple_output(output_path + "p2e_output.png")

        elif extract_mode == "cubemap":
            # detect floor using segmentation model
            ai_predict.run_images(input_images_folder=config.E2C_OUTPUT_FOLDER,
                                  prediction_dir=config.FLOOR_DETECT_OUTPUT_FOLDER + "360" + os.sep)
            cubemap_equi(
                input_folder=config.FLOOR_DETECT_OUTPUT_FOLDER + "360" + os.sep + "comb_mask" + os.sep,
                input_image_prefix=config.E2C_OUTPUT_IMAGE_PREFIX,
                output_image_folder=output_path + "comb_mask" + os.sep,
            )
            correct_360_mask_and_combine(
                mask_path=output_path + "comb_mask" + os.sep + "c2e.png",
                actual_360_image=input_path,
                combined_360_opt_path=output_path + "c2e.png",
            )
            if copy_to_simple_output:
                utils.copy_to_simple_output(output_path + "c2e.png")

        elif extract_mode == "simple_crop":
            # detect floor using segmentation model
            ai_predict.run_images(input_images_folder=config.BUFFER_FOLDER,
                                  prediction_dir=config.FLOOR_DETECT_OUTPUT_FOLDER + "360" + os.sep + "comb_mask" + os.sep)

            # stitch the floor images together
            combine_cropped_360(
                mask_input_folder=config.FLOOR_DETECT_OUTPUT_FOLDER + "360" + os.sep + "comb_mask" + os.sep + "comb_mask" + os.sep,
                image_input_path=input_path,
                output_image_prefix=input_path.split(os.sep)[-1].split(".")[0],
                output_folder=output_path,
            )
            if copy_to_simple_output:
                utils.copy_to_simple_output(output_path + input_path.split(os.sep)[-1].split(".")[0] + ".png")

        print(f"Done! Please check the \"{config.SIMPLE_OUTPUT_FOLDER.split(os.sep)[-2] }\" folder for the output image.")
    else:
        # stitch the perspective images together without running the unet model
        pers_equi(input_dir=config.E2P_OUTPUT_FOLDER, skip_roof=skip_roof)


def predict_all_360_images(input_folder=config.FLOOR_DETECT_360_INPUT_FOLDER,
                           predicted_folder=config.P2E_OUTPUT_FOLDER,
                           output_folder=config.SIMPLE_OUTPUT_FOLDER):

    utils.delete_folder_contents(output_folder)

    for i, file in enumerate(os.listdir(input_folder)):
        print(f"\n\nFile {i + 1}/{len(os.listdir(input_folder))}: {file}\n")
        shutil.copy(input_folder + file, output_folder + file)

        extract_and_predict(input_path=input_folder + file,
                            predict=True,
                            copy_to_simple_output=False,)
        file_wo_ext = file.split(file.split(".")[-1])[0][:-1]

        shutil.copy(predicted_folder + "p2e_output.png", output_folder + file_wo_ext + "_output.png")

    print(f"\n\nDone! Please check the {output_folder.split(os.sep)[-2]} folder for the output images.")


if __name__ == "__main__":
    # pers_equi(input_dir=config.E2P_OUTPUT_FOLDER)
    pers_equi(input_dir=config.FLOOR_DETECT_OUTPUT_FOLDER)
