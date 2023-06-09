import os
import cv2


images_path = "/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/RetinalSegmentation/external_repos/data/CHASEDB/images/"
mask_path = "/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/RetinalSegmentation/external_repos/data/CHASEDB/manual/"

extracted_training_folder = "/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/RetinalSegmentation/external_repos/data/training_data/CHASEDB/"
ext_imgs_folder = extracted_training_folder + "images/"
ext_masks_folder = extracted_training_folder + "masks/"

# iterate through masks folder and find the corresponding image
images_paths = os.listdir(images_path)
masks_paths = os.listdir(mask_path)

images_no_ext = {}
for file in images_paths:
    images_no_ext[file.split(".")[0]] = file

for mask in masks_paths:
    mask_file_path = mask_path + mask
    image_file = images_no_ext[mask.split(".")[0]]
    image_file_path = images_path + image_file

    image = cv2.imread(image_file_path)
    mask = cv2.imread(mask_file_path)

    # save image
    cv2.imwrite(ext_imgs_folder + image_file, image)
    # save mask
    cv2.imwrite(ext_masks_folder + image_file, mask)
