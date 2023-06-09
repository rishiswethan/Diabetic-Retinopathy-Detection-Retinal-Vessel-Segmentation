import os
import json
import shutil
import keyboard

import source.config as config

def load_dict_from_json(file_name):
    with open(file_name) as f:
        d = json.load(f)
        # print(d)

    return d


def save_dict_as_json(file_name, dict, over_write=False):
    if os.path.exists(file_name) and over_write:
        with open(file_name) as f:
            existing_dict = json.load(f)
            # print(existing_dict)

        existing_dict.update(dict)

        print(f"Saving json file name {file_name}\n Data:\n{existing_dict}")
        with open(file_name, 'w') as f:
            json.dump(existing_dict, f)
    else:
        print(f"Saving json file name {file_name}\n Data:\n{dict}")
        with open(file_name, 'w') as f:
            json.dump(dict, f)


# Find a file in a folder with a similar name. Primary use is to take the file even if the enter name without the extension
def find_filename_match(known_filename, directory):
    files_list = os.listdir(directory)
    good_match = None
    for file_name in files_list:
        if known_filename == file_name:
            print("\n\nFile found:", file_name, "; In directory:", directory.split(os.sep)[-2], "\n")
            return os.path.join(directory, file_name)
        elif known_filename.lower() in file_name.lower():
            good_match = os.path.join(directory, file_name)

    if good_match is None:
        print("\n\nNo match found for file name: " + known_filename, "; In directory:", directory.split(os.sep)[-2], "\n")
    elif good_match is not None:
        print("\n\nFound match for file name: " + known_filename, "; In directory:", directory.split(os.sep)[-2], "\n", "Matched file:", good_match.split(os.sep)[-1], "\n")

    return good_match


# recursively delete all content in a folder and the folder itself if chosen
def delete_folder_contents(folder_path, delete_folder=False):
    if not os.path.exists(folder_path) and (not delete_folder):
        os.makedirs(folder_path)

    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

            if delete_folder:
                os.rmdir(folder_path)
    except Exception as e:
        pass


def copy_to_simple_output(target_path, simple_output_path=config.SIMPLE_OUTPUT_FOLDER + f"output.png", verbose=False):
    # Delete all files in the simple output folder and copy the new file to it
    delete_folder_contents(os.path.dirname(simple_output_path), delete_folder=False)
    shutil.copy(target_path, simple_output_path)

    if verbose:
        print("\nCopied file to", simple_output_path)


def return_file_in_folder(folder_path):
    for file in os.listdir(folder_path):
        if not os.path.isdir(folder_path + file):
            print("Found file:", file, "in folder:", folder_path.split(os.sep)[-1], "\n")

            return folder_path + os.sep + file

    return None


def input_no_enter(prompt="", verbose=False):
    print(prompt, end="")
    while True:
        key = keyboard.read_event()
        if key.event_type == keyboard.KEY_DOWN:
            if len(key.name) == 1:
                if verbose:
                    print(key.name)
                else:
                    print()
                return key.name
            else:
                print("Unknown input: " + key.name)
                print("Enter a single character: ", end="")
