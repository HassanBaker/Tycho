from os import listdir
from os.path import isfile, join
from skimage import io
from tqdm import tqdm
import numpy as np
from tools.config import data_dir, TRAIN_DIR

DIR = data_dir + TRAIN_DIR
files = [f for f in listdir(DIR) if isfile(join(DIR, f))]


def load_image(name):
    return io.imread(join(DIR, name))


def convert_to_bw(image):
    return np.sum(image / np.shape(image)[0], axis=2) / 3


def calc_crop_amount(image_file, brightness_limit=0.1):
    img = load_image(image_file)

    img = np.sum(img / 255, axis=2) / 3

    brightness_of_areas = np.zeros(212)

    for i in range(212):
        brightness_of_areas[i] = np.sum(img) / (np.shape(img)[0] - (i * 2)) ** 2
        img[i] = 0
        img[len(img) - 1 - i] = 0
        img[i + 1:-i - 1, i] = 0
        img[i + 1:-i - 1, len(img) - 1 - i] = 0

    for i in range(len(img)):
        if brightness_of_areas[i] > brightness_limit:
            return np.shape(img)[0] - (i * 2)


def approx_total_crop_amount():
    crop_approx = []

    with tqdm(total=len(files)) as pbar:
        for _file in files:
            crop_approx.append(calc_crop_amount(_file))
            pbar.update(1)

    minimum_crop_approx_image = min(crop_approx)
    file_with_min_crop = files[crop_approx.index(minimum_crop_approx_image)]
    print("Minimum: ", minimum_crop_approx_image, " - ", file_with_min_crop)
    print("")

    maximum_crop_approx_image = max(crop_approx)
    file_with_max_crop = files[crop_approx.index(maximum_crop_approx_image)]
    print("Maximum: ", maximum_crop_approx_image, " - ", file_with_max_crop)
    print("")

    standard_deviation = np.std(np.array(crop_approx))
    print("Standard Deviation: ", standard_deviation)
    print("")

    average_crop_amount = sum(crop_approx) / len(crop_approx)
    print("Average Crop Size: ", average_crop_amount)

    np.savetxt("crop_analysis.csv", crop_approx, delimiter=",")


if __name__ == '__main__':
    approx_total_crop_amount()
