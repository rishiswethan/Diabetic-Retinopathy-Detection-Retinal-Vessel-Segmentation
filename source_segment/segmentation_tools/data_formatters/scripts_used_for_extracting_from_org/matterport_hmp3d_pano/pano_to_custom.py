import random

from source_segment.image import equi_perspective

import os
import numpy as np
import shutil
import cv2
import threading
import time


original_dataset_primary_folder = "D:\\matterport_habitat_extracted\\"

input_dir = original_dataset_primary_folder + "pano\\train\\images\\"
input_mask_dir = original_dataset_primary_folder + "pano\\train\\masks\\"
output_dir = original_dataset_primary_folder + "custom_view_angle\\train\\images\\"
output_mask_dir = original_dataset_primary_folder + "custom_view_angle\\train\\masks\\"

input_dir_val = original_dataset_primary_folder + "pano\\val\\images\\"
input_mask_dir_val = original_dataset_primary_folder + "pano\\val\\masks\\"
output_dir_val = original_dataset_primary_folder + "custom_view_angle\\val\\images\\"
output_mask_dir_val = original_dataset_primary_folder + "custom_view_angle\\val\\masks\\"

PERSPECTIVE_SQUARE_SIZE = 1024

MAX_THREADS = 10
num_threads_running = 0

random_fov_list = {}

def extract_multiple_perspectives(
        input_path,
        output_folder,
        make_mask_binary=False,
        view_angle_1=360,
        view_angle_2=330,
        fov_1=60,
        fov_2=100,
        h_increment=8,
        perspective_square_size=1024,
):
    """
    Extracts multiple perspectives from a 360 image
    """
    global num_threads_running, random_fov_list

    if len(random_fov_list) == 0:
        for i in range(0, 360, h_increment):
            random_fov_list[i] = random.randint(fov_1, fov_2)
        print("Random fov list: ", random_fov_list)

    num_threads_running += 1

    def make_masks_binary(output_folder):
        for mask_path in os.listdir(output_folder):
            output_path_mask = output_folder + mask_path
            mask = cv2.imread(output_path_mask, cv2.IMREAD_GRAYSCALE)
            uniq = np.unique(mask)

            if 1 in uniq and 0 in uniq and len(uniq) == 2:
                print("Mask seems to be binary already. Skipping make binary...", uniq)
                continue

            mask[mask > 128] = 1
            mask[mask != 1] = 0
            cv2.imwrite(output_path_mask, mask)

    # Extract perspective images for the main prediction fov. Typically, 90 degrees
    shutil.rmtree(output_folder, ignore_errors=True)
    os.makedirs(output_folder, exist_ok=True)

    output_path = output_folder + "e2p.png"

    for i in np.arange(0, 360, h_increment):
        # print(
        #     f"Extracting image with centre_h: {i}, centre_v: {def_centre_v}, h_fov: {h_field_of_view}, v_fov: {v_field_of_view}")

        # ensure that the input image is RGB
        input_img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if len(input_img.shape) == 2:
            input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(input_path, input_img)

        # extract the specified perspective
        center_v = random.randint(view_angle_2, view_angle_1)
        fov = random_fov_list[i]
        equi_perspective(input_path,
                         {
                             "horizontal_viewing_angle": i,
                             "vertical_viewing_angle": center_v,
                             "h_field_of_view": fov,
                             "v_field_of_view": fov,
                             "output_image_size": [perspective_square_size, perspective_square_size],
                             "image_rotate": 0
                         },
                         output_path=output_path)

    if make_mask_binary:
        make_masks_binary(output_folder)

    num_threads_running -= 1


def run_all_scenes(input_dir, output_dir, make_mask_binary=False, ext=".png"):
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)
    for i, folder in enumerate(os.listdir(input_dir)):
        if not os.path.isdir(input_dir + folder):
            continue

        for j, image_name in enumerate(os.listdir(input_dir + folder)):
            if not image_name.endswith(ext):
                continue

            file_path = input_dir + folder + os.sep + image_name
            # 00006-HkseAnWCgqk_height-0.00_pt0.png
            # images/00006-HkseAnWCgqk/height-0.00_pt0
            output_folder = output_dir + image_name.split("_")[0] + os.sep + image_name.split("_")[1] + "_" + image_name.split("_")[2].split(ext)[0] + os.sep

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            print(f"Processing image {i} {j} {folder}/{image_name}")
            while num_threads_running >= MAX_THREADS:
                time.sleep(0.5)

            # extract_multiple_perspectives(file_path, output_folder, make_mask_binary)
            threading.Thread(target=extract_multiple_perspectives, args=(
                file_path,
                output_folder,
                make_mask_binary
            )).start()


if __name__ == "__main__":
    run_all_scenes(input_dir, output_dir)
    run_all_scenes(input_mask_dir, output_mask_dir, make_mask_binary=True, ext=".png")
    run_all_scenes(input_dir_val, output_dir_val)
    run_all_scenes(input_mask_dir_val, output_mask_dir_val, make_mask_binary=True, ext=".png")
