import numpy as np
import matplotlib.pyplot as plt
import webcolors
import cv2
import os

import habitat_sim
from habitat_sim.utils.settings import make_cfg
import csv
import json
import shutil
import traceback


label_lists = {
    'floor': [
        'floor',
        # 'rug'
    ],
    'wall': ['wall'],
    'ceiling': ['ceiling'],
}
sub_name_to_main_name = {
    'floor': 'floor',
    'rug': 'floor',
    'wall': 'wall',
    'ceiling': 'ceiling',
}
label_mask_output_numbers = {
    'floor': 1,
    'wall': 2,
    'ceiling': 3,
}


# For viewing the extractor output
def display_sample(sample):
    img = sample["rgb"]
    semantic = sample["semantic"]
    # print(sample.keys())
    # print("semantic shape: ", semantic.shape)

    arr = [img, semantic]
    titles = ["rgba", "semantic"]
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)

    plt.show()


def get_mask_vals_of_label(json_dict, label_lists=label_lists, label_mask_output_numbers=label_mask_output_numbers):
    # get the "class_id" of the label
    label_to_id = {}
    classes_json = json_dict["classes"]

    # get the "id" of the label. This is the value that is used in the semantic image
    objects_json = json_dict["objects"]
    class_name_to_mask_ids = {}  # key: class_name, value: list of mask ids
    for obj in objects_json:
        class_name = obj["class_name"]  # this is the name of the label
        mask_num = obj["id"]  # this is the value that is used in the semantic image

        # a label can have multiple names. For example, the label "floor" can also be called "rug"
        if class_name in list(sub_name_to_main_name.keys()):
            if sub_name_to_main_name[class_name] not in list(class_name_to_mask_ids.keys()):
                class_name_to_mask_ids[sub_name_to_main_name[class_name]] = [mask_num]
            elif mask_num not in class_name_to_mask_ids[sub_name_to_main_name[class_name]]:
                class_name_to_mask_ids[sub_name_to_main_name[class_name]].append(mask_num)

    return class_name_to_mask_ids


def get_desired_mask(semantic_image, class_name_to_mask_ids, label_mask_output_numbers=label_mask_output_numbers):
    # semantic_image is a 2d numpy array
    # class_name_to_mask_ids is a dictionary with key: class_name, value: list of mask ids
    # label_mask_output_numbers is a dictionary with key: class_name, value: the desired mask value for that class

    # create a mask with the same shape as the semantic image
    mask = np.zeros(semantic_image.shape, dtype=np.uint8)

    for class_name, mask_ids in class_name_to_mask_ids.items():
        for mask_id in mask_ids:
            mask[semantic_image == mask_id] = label_mask_output_numbers[class_name]

    return mask


def get_images_and_masks(mesh_semantic_ply, label_lists=label_lists, label_mask_output_numbers=label_mask_output_numbers):
    from habitat_sim.utils.data import ImageExtractor

    # mesh_semantic_ply is the path to the mesh_semantic.ply file
    # label_lists is a dictionary with key: class_name, value: list of labels
    # label_mask_output_numbers is a dictionary with key: class_name, value: the desired mask value for that class

    # get the json file
    scene_info_json = mesh_semantic_ply.replace("mesh_semantic.ply", "info_semantic.json")
    json_list = dict(json.load(open(scene_info_json)))

    # get the mask ids for each class
    class_name_to_mask_ids = get_mask_vals_of_label(json_list, label_lists, label_mask_output_numbers)
    print("class_name_to_mask_ids", class_name_to_mask_ids)

    # get the extractor
    extractor = ImageExtractor(
        mesh_semantic_ply,
        img_size=(512, 512),
        output=["rgba", "depth", "semantic"],
        shuffle=False,
        use_caching=False,
    )
    extractor.set_mode('full')
    rbg_images, masks = [], []
    for i, sample in enumerate(extractor):
        # print(sample.keys())
        print(i + 1, "/", len(extractor))

        # get the image and mask
        rgb_image = sample["rgba"][..., :3]
        semantic_image = sample["semantic"]
        # display_sample({"rgb": rgb_image, "semantic": semantic_image})

        mask = get_desired_mask(semantic_image, class_name_to_mask_ids, label_mask_output_numbers)

        rbg_images.append(rgb_image)
        masks.append(mask)

        # display_sample({"rgb": rgb_image, "semantic": mask})

    extractor.close()

    return rbg_images, masks


def get_images_from_all_folders(main_folder, target_folder):
    # main_folder is the path to the folder that contains all the folders
    # target_folder is the name of the folder that will save all the extracted images and masks

    # get the list of folders
    folders = os.listdir(main_folder)

    # iterate through each folder
    folder_cnt = 1
    total_cnt = 1
    for fold_i, folder in enumerate(folders):
        print("\n\n----->", fold_i, folder)

        # get the path to the mesh_semantic.ply file. apt0/habitat/mesh_semantic.ply
        mesh_semantic_ply = os.path.join(main_folder, folder, "habitat", "mesh_semantic.ply")

        try:
            # get the images and masks
            rbg_images, masks = get_images_and_masks(mesh_semantic_ply)
        except:
            print("---------------------ERROR IN FOLDER--------------------", folder)
            traceback.print_exc()
            continue

        # create the folder for the images and masks
        if not os.path.exists(os.path.join(target_folder, "images", str(folder))):
            os.makedirs(os.path.join(target_folder, "images", str(folder)))
        else:
            print("folder already exists", os.path.join(target_folder, "images", str(folder)))
            shutil.rmtree(os.path.join(target_folder, "images", str(folder)), ignore_errors=False, onerror=None)

        if not os.path.exists(os.path.join(target_folder, "masks", str(folder))):
            os.makedirs(os.path.join(target_folder, "masks", str(folder)))
        else:
            print("folder already exists", os.path.join(target_folder, "masks", str(folder)))
            shutil.rmtree(os.path.join(target_folder, "masks", str(folder)), ignore_errors=False, onerror=None)

        cnt = 1
        # save the images and masks in the target folder
        for i, (rgb_image, mask) in enumerate(zip(rbg_images, masks)):
            print("---------------------")
            print("folder", folder, "cnt", cnt, "total", total_cnt)
            print("---------------------")

            # save the image
            image_name = str(folder) + "_" + str(cnt) + ".jpg"
            image_path = os.path.join(target_folder, "images", str(folder), image_name)
            cv2.imwrite(image_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

            # save the mask
            mask_name = str(folder) + "_" + str(cnt) + ".png"
            mask_path = os.path.join(target_folder, "masks", str(folder), mask_name)
            cv2.imwrite(mask_path, mask)

            cnt += 1
            total_cnt += 1

        folder_cnt += 1


if __name__ == "__main__":
    get_images_from_all_folders(
        main_folder="/home/rishi/programming/AI/experiments/datasets_extractor/data/Replica/",
        target_folder="/home/rishi/programming/AI/experiments/datasets_extractor/extracted_data/replica/"
    )

