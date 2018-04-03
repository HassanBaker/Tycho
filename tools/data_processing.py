import random
from os import listdir
from os.path import isfile, join

import numpy as np
from skimage import io
from skimage.transform import rescale
from skimage.transform import rotate

from tools.config import root_dir, labels_df

"""
Contains data processing classes:
    image_augmenter
    image_data

Dependencies:
    tools.config
"""


# noinspection PyMethodMayBeStatic,SpellCheckingInspection
class image_augmenter:
    """
    A class used to agument images.
    Contains methods for cropping, rotating, downscaling, and alligning.

    Contains some high level augmentations that carry out an ordered set of augmentations:
    Higher level augmentation methods:
        augment()   -> a list of 16 augments of one input image
        reduce()    -> a single image, that equates to the image at index 0, which augment() returns

    """

    def crop_image(self, image, cropped_size=208):

        image_size = np.shape(image)[0]

        if cropped_size > image_size:
            raise Exception("cropped_size cannot be greater than the image size")

        crop_amount = round((image_size - cropped_size) / 2)
        return image[crop_amount:-crop_amount, crop_amount:-crop_amount]

    def downscale(self, image, ds_scale=1 / 3):
        return rescale(image, ds_scale)

    def rotate__image_45(self, image):
        return rotate(image, 45)

    def allign_1(self, image):
        return rotate(image, 90)

    def allign_2(self, image):
        return rotate(image, 270)

    def allign_3(self, image):
        return rotate(image, 180)

    def flip_horz(self, image):
        return image[:, ::-1]

    def get_top_left(self, image):
        return image[:45, :45]

    def get_top_right(self, image):
        return image[:45, 24:]

    def get_bottom_left(self, image):
        return image[24:, :45]

    def get_bottom_right(self, image):
        return image[24:, 24:]

    def rotation_segment(self, image):
        return [image, self.rotate__image_45(image)]

    def flipping_segment(self, images_list):
        return images_list + [self.flip_horz(image) for image in images_list]

    def crop_segment(self, images_list):
        return (self.crop_image(image) for image in images_list)

    def downscale_segment(self, images_list):
        return (self.downscale(image) for image in images_list)

    def get_all_corners(self, images_list):
        list_of_corners = []
        for image in images_list:
            list_of_corners.append(
                self.get_top_left(image))

            list_of_corners.append(
                self.allign_1(
                    self.get_top_right(image)))

            list_of_corners.append(
                self.allign_2(
                    self.get_bottom_left(image)))

            list_of_corners.append(
                self.allign_3(
                    self.get_bottom_right(image)))

        return list_of_corners

    def augment(self, image):
        ROTATED = self.rotation_segment(image)
        FLIPPED = self.flipping_segment(ROTATED)
        CROPPED = self.crop_segment(FLIPPED)
        DOWNSCALED = self.downscale_segment(CROPPED)
        CORNERS = self.get_all_corners(DOWNSCALED)
        return CORNERS

    def reduce(self, image):
        CROPPED = self.crop_image(image)
        DOWNSCALED = self.downscale(CROPPED)
        CORNER = self.get_top_left(DOWNSCALED)
        return CORNER


class image_data:
    """
    A class used to load and process image data.

    __init__:
        dir_name    - directory name (string)
        type        - if "TEST", does not load labels (test labels do not exist)
        augments    - "REDUCE":
                        runs reduce() in image_augmenter
                      "CONCAT":
                        runs augment() in image_augmenter
                      "SELECT_ONE"
                        runs augment() in image_augmenter, and selects a random image from the 16 returned

    """

    def __init__(self, dir_name, type="TRAIN", augment="REDUCE"):
        self.directory = root_dir + dir_name
        self.file_names = [f for f in listdir(self.directory) if isfile(join(self.directory, f))]
        self.TYPE = type
        if self.TYPE != "TEST":
            self.labels_df = labels_df
        self._augmenter = image_augmenter()
        self._cursor = 0
        self._augment = augment
        # print(dir_name, "read", "- Init Complete")

    def shuffle(self):
        random.shuffle(self.file_names)
        # print("Shuffle - Complete\n")

    def get_image_labels(self, file_name):
        labels = self.labels_df[self.labels_df["GalaxyID"] == int(file_name[:-4])]

        labels.drop("GalaxyID", axis=1, inplace=True)

        labels = np.array(labels.get_values()).reshape(labels.shape[1])

        return labels

    def reset(self):
        random.shuffle(self.file_names)
        self._cursor = 0

    def next_batch(self, batch_size):
        labels_list = []
        images_list = []
        for i in range(batch_size):
            file_name = self.file_names[self._cursor]

            if self.TYPE != "TEST":
                labels_list.append(self.get_image_labels(file_name))
            else:
                labels_list.append(file_name[:-4])

            image = self._load_image(file_name)

            if self._augment is not None:
                if self._augment == "CONCAT":
                    image = self._augmenter.augment(image)
                elif self._augment == "REDUCE":
                    image = self._augmenter.reduce(image)
                elif self._augment == "SELECT_ONE":
                    images = self._augmenter.augment(image)
                    image = random.choice(images)

            images_list.append(image)

            self._cursor += 1
            if self._cursor == len(self.file_names):
                self.shuffle()
                self._cursor = 0

        if self._augment == "CONCAT":
            return np.array(images_list).reshape(batch_size * 16, 45, 45, 3), np.array(labels_list)
        else:
            return np.array(images_list), np.array(labels_list)

    def _load_image(self, filename):
        return io.imread(join(self.directory, filename)) / 255
