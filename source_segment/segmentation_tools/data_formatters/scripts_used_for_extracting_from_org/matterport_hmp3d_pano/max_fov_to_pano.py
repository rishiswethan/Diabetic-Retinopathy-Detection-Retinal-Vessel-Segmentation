import shutil

import source_segment.tools_360.multi_p2e as m_P2E
import cv2
import os
import numpy as np
import threading
import matplotlib.pyplot as plt
import time

import source_segment.config as cf

# original_dataset_primary_folder = cf._ORIGINAL_DATASET_PRIMARY_FOLDER + "hm3d_pano\\max_fov\\"
original_dataset_primary_folder = "D:\\matterport_habitat_extracted\\"

# input_dir = original_dataset_primary_folder + "max_fov\\train\\images\\"
# input_mask_dir = original_dataset_primary_folder + "max_fov\\train\\masks\\"
# output_dir = original_dataset_primary_folder + "pano\\train\\images\\"
# output_mask_dir = original_dataset_primary_folder + "pano\\train\\masks\\"
# buffer_folder = original_dataset_primary_folder + "pano\\buffer\\"

input_dir = original_dataset_primary_folder + "max_fov\\val\\images\\"
input_mask_dir = original_dataset_primary_folder + "max_fov\\val\\masks\\"
output_dir = original_dataset_primary_folder + "pano\\val\\images\\"
output_mask_dir = original_dataset_primary_folder + "pano\\val\\masks\\"
buffer_folder = original_dataset_primary_folder + "pano\\buffer\\"


fov = 160.0
out_height = 1500
out_width = 3000
target_mask_number = 1  # floor is 1

MAX_THREADS = 10
num_threads_running = 0

# create the buffer folder if it doesn't exist
shutil.rmtree(buffer_folder, ignore_errors=True)
os.makedirs(buffer_folder)

def convert_to_pano(input_dir, output_path, fov=fov, out_height=out_height, out_width=out_width, ext=".jpg"):
    global num_threads_running

    num_threads_running += 1
    input_images = []
    img_attributes = []
    mask_images = []
    empty_image = None

    if len(os.listdir(input_dir)) == 0:
        num_threads_running -= 1
        return False

    # empty the buffer folder
    sub_buffer_folder = buffer_folder + input_dir.split(os.sep)[-3] + os.sep + input_dir.split(os.sep)[-2] + os.sep + ext.split(".")[1] + os.sep
    if os.path.exists(sub_buffer_folder):
        shutil.rmtree(sub_buffer_folder)
    os.makedirs(sub_buffer_folder)

    for file_ in os.listdir(input_dir):
        file = input_dir + file_
        buffer_folder_file = sub_buffer_folder + os.sep + file_

        shutil.copy(file, buffer_folder_file)
        file = buffer_folder_file

        if os.path.isdir(file):
            continue

        # replace 2d bw images with 3d bw images
        if file.endswith(".png") and not file.endswith("empty_image.png") and not file.endswith("white_image.png"):
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 2:
                img = np.stack((img,)*3, axis=-1)

                img[img == target_mask_number] = 255
                img[img != 255] = 0
                cv2.imwrite(file, img)

        if empty_image is None:
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            empty_image = np.zeros(img.shape, dtype=np.uint8)
            empty_image.fill(0)
            # save the empty image
            cv2.imwrite(output_dir + "empty_image.png", empty_image)

            white_image = np.zeros(img.shape, dtype=np.uint8)
            white_image.fill(255)
            # save the white image
            cv2.imwrite(output_dir + "white_image.png", white_image)

        # print(file)

        v = 0.0
        h = float(file.split("angle")[1].split(ext)[0])
        vfov = fov
        hfov = fov
        # print("hfov", hfov, "h", h, "v", v)

        img_attributes.append([hfov, h, v])
        input_images.append(file)

    equ = m_P2E.Perspective(input_images,
                            img_attributes)
    print("\nConverting to equirectangular...")
    img_ = equ.get_equirec(out_height, out_width)
    # print("equ shape", img_.shape)

    if os.path.exists(output_path):
        os.remove(output_path)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Writing to", output_path, " with shape ", img_.shape)
    cv2.imwrite(output_path, img_)
    if ext == '.png':
        # convert to grayscale
        img = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img[img > 128] = 255
        img[img != 255] = 0
        cv2.imwrite(output_path, img)

    num_threads_running -= 1
    return True


shutil.rmtree(output_dir, ignore_errors=True)
shutil.rmtree(output_mask_dir, ignore_errors=True)
for i, folder in enumerate(os.listdir(input_dir)):
    if not os.path.isdir(input_dir + folder):
        continue

    for j, folder_pt in enumerate(os.listdir(input_dir + folder)):
        if not os.path.isdir(input_dir + folder + os.sep + folder_pt):
            continue

        output_name = folder + "_" + folder_pt + ".png"
        output_path = output_dir + folder + os.sep + output_name
        while num_threads_running > MAX_THREADS:
            time.sleep(0.1)

        threading.Thread(target=convert_to_pano, args=(
            input_dir + folder + os.sep + folder_pt + os.sep,
            output_path,
            fov,
            out_height,
            out_width,
            ".jpg"
        )).start()

    if not os.path.isdir(input_mask_dir + folder):
        continue

    for j, folder_pt in enumerate(os.listdir(input_mask_dir + folder)):
        if not os.path.isdir(input_mask_dir + folder + os.sep + folder_pt):
            continue

        output_name = folder + "_" + folder_pt + ".png"
        output_path = output_mask_dir + folder + os.sep + output_name

        while num_threads_running > MAX_THREADS:
            time.sleep(0.1)

        threading.Thread(target=convert_to_pano, args=(
            input_mask_dir + folder + os.sep + folder_pt + os.sep,
            output_path,
            fov,
            out_height,
            out_width,
            ".png"
        )).start()


# delete the empty image and white image and buffer folder
while num_threads_running > 0:
    print("Waiting for threads to finish", num_threads_running)
    time.sleep(0.5)

if os.path.exists(output_dir + "empty_image.png"):
    os.remove(output_dir + "empty_image.png")
if os.path.exists(output_dir + "white_image.png"):
    os.remove(output_dir + "white_image.png")
if os.path.exists(output_mask_dir + "empty_image.png"):
    os.remove(output_mask_dir + "empty_image.png")
if os.path.exists(output_mask_dir + "white_image.png"):
    os.remove(output_mask_dir + "white_image.png")

shutil.rmtree(buffer_folder, ignore_errors=True)
