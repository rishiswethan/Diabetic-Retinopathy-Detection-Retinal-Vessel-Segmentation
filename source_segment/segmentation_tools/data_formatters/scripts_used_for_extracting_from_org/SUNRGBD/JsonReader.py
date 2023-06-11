import json
import csv
import os.path
import sys
import cv2
import numpy as np
from PIL import Image
import traceback
import os

all_lables = []

with open('all_labels.txt') as label_names:
    for line in label_names.readlines():
        line = line.strip("\n")
        all_lables.append(line)

search_names = [
    "floor",
    "floor_sticker",
    "shower_floor",
    "floor mat",
    "dark_floor",
    "floor_mat",
    "floor_mats",
    "office_carpet",
    "carpet_area",
    "carpet_role",
    "carpet",
    "gray_carpet",
    "another_carpet",
    "run_or_carpet",
    "other_room_carpet",
    "carpets",
]
replacement_name = "floor"

print("checking", search_names)
image_names = open("image_names.txt", "r")
with open('annotation_names.txt') as annotation_names:
    for annotation_file in annotation_names.readlines():
        annotation_file = annotation_file.strip("\n")
        image_file = image_names.readline()
        image_file = image_file.strip("\n")
        
        """
        if not 'NYU1230' in image_file:
            # print("Skipping", image_file)
            continue
        """
        
        img = cv2.imread(image_file)
        polygons = []
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        print("File Name = ", annotation_file)
        with open(annotation_file) as json_file:
            data = json.load(json_file)
            # print(data)
            # print(data.keys())
            # print(data['name'])
            # print(data['frames'][0]['polygon'][0].keys())
            # print("x: ", data['frames'][0]['polygon'][0]['x'])
            # print("y: ", data['frames'][0]['polygon'][0]['y'])
            # if "XYZ" in data['frames'][0]['polygon'][0].keys():
            # print("XYZ: ", data['frames'][0]['polygon'][0]['XYZ'])
            print("undet poly", len(data['frames'][0]['polygon']))
            edited_count = 0
            for x in range(len(data['frames'][0]['polygon'])):
                if len(data['objects']) < data['frames'][0]['polygon'][x]['object']:
                    continue
                if type(data['frames'][0]['polygon'][x]['x']) != list:
                    continue
                try:
                    current_object = data['objects'][data['frames'][0]['polygon'][x]['object']]['name']
                    current_object = str.lower(current_object).strip()
                    # print("x: ", data['frames'][0]['polygon'][x]['x'])
                    # print("y: ", data['frames'][0]['polygon'][x]['y'])
                    # print(current_object + "|")

                    if not (current_object in search_names):
                        continue
                    
                    
                    print("found--> ", current_object)
                    # print("floor", len(data['frames'][0]['polygon'][x]['x']))
                    polygon = []
                    for y in range(len(data['frames'][0]['polygon'][x]['x'])):
                        points = []
                        points.append(data['frames'][0]['polygon'][x]['x'][y])
                        points.append(data['frames'][0]['polygon'][x]['y'][y])
                        polygon.append(points)
                    if len(polygon) > 2:
                        polygons.append(polygon)
                except:
                    traceback.print_exc()

            print("polylen", len(polygons))
            masks = []
            for polygon in polygons:
                pts = np.array(polygon, np.int32)
                pts = pts.reshape((-1, 1, 2))
                # print(polygon)
                mask = cv2.fillPoly(mask, [pts], 255)
                masks.append(mask)
                # Image.fromarray(mask).show()

            combined_mask = None
            for mask in masks:
                if combined_mask is None:
                    combined_mask = mask.copy()
                else:
                    combined_mask = np.add(combined_mask, mask)
                    combined_mask[combined_mask > 255] = 255
                # Image.fromarray(mask).show()
                
            mask = combined_mask
                                
            # if combined_mask is not None:
            #     Image.fromarray(combined_mask).show()
            
            # print(image_file)
            image_file_parts = image_file.split("/")
            new_image_file_name = ""
            for x in range(len(image_file_parts) - 2):
                new_image_file_name = new_image_file_name + os.sep + image_file_parts[x]
            new_image_file_name = new_image_file_name + os.sep + replacement_name
            new_image_file_name = new_image_file_name[1:]
            if not os.path.exists(new_image_file_name):
                os.mkdirs(new_image_file_name)
            new_image_file_name = new_image_file_name + os.sep + image_file_parts[len(image_file_parts) - 1]
            new_image_file_name = new_image_file_name.replace(".jpg", ".png")
            print(new_image_file_name)
            # Image.fromarray(mask).save(new_image_file_name)
            if os.path.exists(new_image_file_name):
                print("removing", new_image_file_name)
                os.remove(new_image_file_name)
            if mask is not None:
                cv2.imwrite(new_image_file_name, mask)
                print("saved", new_image_file_name)
            print(data['frames'][0]['polygon'][0]['object'], ": ", all_lables[data['frames'][0]['polygon'][0]['object']])

