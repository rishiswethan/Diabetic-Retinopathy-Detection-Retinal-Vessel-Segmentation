import os

rgb_directory = "data/rgb"
semantic_pretty_directory = "data/semantic_pretty"
semantic_directory = "data/semantic"

rgb_files = os.listdir(rgb_directory)
semantic_pretty_files = os.listdir(semantic_pretty_directory)
semantic_files = os.listdir(semantic_directory)

rgb_file_names = open("rgb_file_names.txt", "w")
semantic_pretty_file_names = open("semantic_pretty_file_names.txt", "w")
semantic_file_names = open("semantic_file_names.txt", "w")

for x in range(len(rgb_files)):
    if rgb_files[x] == ".gitkeep":
        continue
    rgb_file_names.write(rgb_directory + "/" + rgb_files[x] + "\n")
rgb_file_names.close()

for x in range(len(semantic_pretty_files)):
    if semantic_pretty_files[x] == ".gitkeep":
        continue
    semantic_pretty_file_names.write(semantic_pretty_directory + "/" + semantic_pretty_files[x] + "\n")
semantic_pretty_file_names.close()

for x in range(len(semantic_files)):
    if semantic_files[x] == ".gitkeep":
        continue
    semantic_file_names.write(semantic_directory + "/" + semantic_files[x] + "\n")

semantic_file_names.close()
