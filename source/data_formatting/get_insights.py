import source.config as cf

import os

trainings_folder = cf.DATA_FOLDERS['training_data']

dataset_counts_train = {}
dataset_counts_test = {}
dataset_counts_classwise = {}
for dataset in os.listdir(trainings_folder):
    dataset_folder = trainings_folder + dataset + os.sep

    dataset_counts_train[dataset.split("-")[0]] = len(os.listdir(dataset_folder + 'train' + os.sep))
    dataset_counts_test[dataset.split("-")[0]] = len(os.listdir(dataset_folder + 'test' + os.sep))

    for class_folder in os.listdir(dataset_folder + 'all' + os.sep):
        if dataset_counts_classwise.get(class_folder) is None:
            dataset_counts_classwise[class_folder] = len(os.listdir(dataset_folder + 'all' + os.sep + class_folder))
        else:
            dataset_counts_classwise[class_folder] += len(os.listdir(dataset_folder + 'all' + os.sep + class_folder))

print("Train counts:")
print(dataset_counts_train)

print("Test counts:")
print(dataset_counts_test)

print("Classwise counts:")
print(dataset_counts_classwise)

# create a bar plot for the train counts and test counts
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 10))

plt.bar(dataset_counts_train.keys(), dataset_counts_train.values(), width=0.5, color='b', align='center')
plt.bar(dataset_counts_test.keys(), dataset_counts_test.values(), width=0.5, color='r', align='center')
plt.xticks(rotation=90)
plt.ylabel("Count")
plt.title("Train and Test Counts")
plt.legend(['Train', 'Test'])
plt.savefig("train_test_counts.png")
plt.close()

# create a bar plot for the classwise counts
plt.figure(figsize=(10, 13))

plt.bar(dataset_counts_classwise.keys(), dataset_counts_classwise.values(), width=0.5, color='b', align='center')
plt.xticks(rotation=90)
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Classwise Counts")
plt.savefig("classwise_counts.png")
plt.close()
