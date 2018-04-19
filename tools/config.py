import pandas as pd
from os.path import join

"""
This file contains useful variables and configurations that are extensively used throughout studies and training
"""

data_dir = "directory where all the data is stored"
TRAIN_DIR = "training"
VALIDATION_DIR = "validation"
FULL_TRAIN_DIR = "images_training_rev1"
TEST_DIR = "images_test_rev1"
labels_df = pd.read_csv(join(data_dir, "training_solutions_rev1.csv"))

solutions_dir = "The directory where you want your solution files stored"
log_dir = "The directory where you want your TenorBoard log files stored"
save_dir = "The directory where you want your TensorFlow sessions to be saved"

labels = ["GalaxyID",
          "Class1.1",
          "Class1.2",
          "Class1.3",
          "Class2.1",
          "Class2.2",
          "Class3.1",
          "Class3.2",
          "Class4.1",
          "Class4.2",
          "Class5.1",
          "Class5.2",
          "Class5.3",
          "Class5.4",
          "Class6.1",
          "Class6.2",
          "Class7.1",
          "Class7.2",
          "Class7.3",
          "Class8.1",
          "Class8.2",
          "Class8.3",
          "Class8.4",
          "Class8.5",
          "Class8.6",
          "Class8.7",
          "Class9.1",
          "Class9.2",
          "Class9.3",
          "Class10.1",
          "Class10.2",
          "Class10.3",
          "Class11.1",
          "Class11.2",
          "Class11.3",
          "Class11.4",
          "Class11.5",
          "Class11.6"
          ]