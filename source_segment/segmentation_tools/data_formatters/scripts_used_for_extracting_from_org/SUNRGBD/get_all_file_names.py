import sys
import os

currentDir = "SUNRGBD"
# next set of directories (kv1, kv2, ...)
dataset_types = os.listdir(currentDir)
for x in range(len(dataset_types)):
    dataset_types[x] = currentDir + "/" + dataset_types[x]
dataset_types_refined = []

# remove anything that is not a directory
for x in range(len(dataset_types)):
    if os.path.isdir(dataset_types[x]):
        dataset_types_refined.append(dataset_types[x])

# next set of directories which is also the name of each dataset.
all_datasets = []
for x in range(len(dataset_types_refined)):
    current_list = os.listdir(dataset_types_refined[x])
    for y in range(len(current_list)):
        current_list[y] = dataset_types_refined[x] + "/" + current_list[y]
        if os.path.isdir(current_list[y]):
            all_datasets.append(current_list[y])

# next set of directories (after this for almost each dataset, we reach the point where the data of individual images
# is stored except for sun3D which has a slightly different file structure
all_folders = []
for x in range(len(all_datasets)):
    current_list = os.listdir(all_datasets[x])
    for y in range(len(current_list)):
        current_list[y] = all_datasets[x] + "/" + current_list[y]
        if os.path.isdir(current_list[y]):
            all_folders.append(current_list[y])

# Adding the folders of the Sun3D dataset
all_folders_refined = []
for x in range(len(all_folders)):
    if "sun3ddata" in all_folders[x]:
        current_list1 = os.listdir(all_folders[x])
        for y in range(len(current_list1)):
            current_list2 = os.listdir(all_folders[x] + "/" + current_list1[y])
            for z in range(len(current_list2)):
                all_folders_refined.append(all_folders[x] + "/" + current_list1[y] + "/" + current_list2[z])
    else:
        all_folders_refined.append(all_folders[x])

# separating image and annotation2D file names and writing it in the respective files.
image_files = []
annotation_files = []
image_names = open("image_names.txt", "w")
annotation_names = open("annotation_names.txt", "w")

for x in range(len(all_folders_refined)):
    image_folder_name = all_folders_refined[x] + "/image/"
    annotation_file_name = all_folders_refined[x] + "/annotation2Dfinal/index.json"
    image_file_name = image_folder_name + os.listdir(image_folder_name)[0]
    image_names.write(image_file_name + "\n")
    annotation_names.write(annotation_file_name + "\n")

image_names.close()
annotation_names.close()
