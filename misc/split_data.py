import os
from os import listdir
from os.path import isfile, join
from random import shuffle
from shutil import copyfile, rmtree

DIR = "/home/hassan/fyp/data/images_training_rev1"
validation_set_dir = "/home/hassan/fyp/data/validation"
training_set_dir = "/home/hassan/fyp/data/training"

try:
    rmtree(validation_set_dir)
except Exception as e:
    pass

try:
    rmtree(training_set_dir)
except Exception as e:
    pass

files = [f for f in listdir(DIR) if isfile(join(DIR, f))]
# print(files[0])
shuffle(files)
# print(files[0])

ten_percent_of_num__of_files = round(len(files) * .1)
validation_files = files[:ten_percent_of_num__of_files]
training_files = files[ten_percent_of_num__of_files:]


def move_files_to_new_dir(src, list_of_files, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for filename in list_of_files:
        copyfile(join(src, filename), join(dst, filename))
    print("Copying complete")


move_files_to_new_dir(DIR, validation_files, validation_set_dir)
move_files_to_new_dir(DIR, training_files, training_set_dir)