from PIL import Image
import os

def recursively_change_file_types(folder):
    for file in os.listdir(folder):
        if os.path.isdir(folder + file):
            recursively_change_file_types(folder + file + "\\")
        else:
            if file.endswith(".ppm"):
                im = Image.open(folder + file)
                im.save(folder + file.replace(".ppm", ".jpg"))
                print("Converted file: " + folder + file)
                # delete ppm file
                os.remove(folder + file)


def crop_x_pixels_from_all_side_of_image(folder, x):
    for file in os.listdir(folder):
        if os.path.isdir(folder + file):
            crop_x_pixels_from_all_side_of_image(folder + file + "\\", x)
        else:
            if file.endswith(".ppm"):
                im = Image.open(folder + file)
                im = im.crop((x, x, im.width - x, im.height - x))
                im.save(folder + file)
                print("Cropped file: " + folder + file)


# recursively_change_file_types("C:\\MySSD\\Programming\\AI\\ClientProjects\\develop\\FloorReplace\\data\\original_data\\nyu\\MaskedImagesFilteredWall\\")
# crop_x_pixels_from_all_side_of_image("C:\\MySSD\\Programming\\AI\\ClientProjects\\develop\\FloorReplace\\data\\original_data\\nyu\\MaskedImagesFilteredWall\\", 10)
# crop_x_pixels_from_all_side_of_image("C:\\MySSD\\Programming\\AI\\ClientProjects\\develop\\FloorReplace\\data\\original_data\\nyu\\MaskedImagesFilteredFloor\\", 10)
# crop_x_pixels_from_all_side_of_image("C:\\MySSD\\Programming\\AI\\ClientProjects\\develop\\FloorReplace\\data\\original_data\\nyu\\originalImages\\", 10)
