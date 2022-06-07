import rasterio as rio
import os
from patchify import patchify
import cv2
import glob
import csv
import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd

RANDOM_STATE = 42

MASK_ALGORITHM = 'Kumar-Roy'  # TODO: Mask algorithms need to be defined and naming conventions must be followed

ROOT_DIR = '../../FireDetectionProcesses/data'
IMG_DIR = ROOT_DIR + "images/"
MASK_DIR = ROOT_DIR + "masks/"

PATCH_SIZE = 256
BANDS: tuple = (8, 3, 2)

for path, subdir, files in os.walk(IMG_DIR):
    dir_name = path.split(os.path.sep)[-1]
    images = os.listdir(path)
    for k, image_name in enumerate(images):
        if image_name.endswith(".tif"):
            image = rio.open(path + "/" + image_name).read(BANDS).transpose((1, 2, 0))
            SIZE_X = image.shape[0] // PATCH_SIZE * PATCH_SIZE
            SIZE_Y = image.shape[1] // PATCH_SIZE * PATCH_SIZE
            image = image[:SIZE_X, :SIZE_Y, :]
            patches_img = patchify(image, (256, 256, len(BANDS)), step=256)

            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i, j, :, :][0]
                    cv2.imwrite(ROOT_DIR + "256_patches/images/"+image_name +
                                "patch_" + str(i) + str(j) + ".tif", single_patch_img)


for path, subdir, files in os.walk(MASK_DIR):
    dir_name = path.split(os.path.sep)[-1]
    masks = os.listdir(path)
    for k, mask_name in enumerate(masks):
        if mask_name.endswith(".tif"):
            mask = rio.open(path+"/"+mask_name).read()
            SIZE_X = mask.shape[0] // PATCH_SIZE * PATCH_SIZE
            SIZE_Y = mask.shape[1] // PATCH_SIZE * PATCH_SIZE
            mask = mask[:SIZE_X, :SIZE_Y, :]
            patches_mask = patchify(mask, (256, 256), step=256)

            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):
                    single_patch_img = patches_mask[i, j, :, :][0]
                    cv2.imwrite(ROOT_DIR + "256_patches/images/"+mask_name +
                                "patch_" + str(i) + str(j) + ".tif", single_patch_img)

IMAGES_PATH = '../../FireDetectionProcesses/data/256_patches/images'
MASKS_PATH = '../../FireDetectionProcesses/data/256_patches/masks'
OUTPUT = '../../FireDetectionProcesses/data'

IMAGES_DATAFRAME = os.path.join(OUTPUT, 'images_masks.csv')

TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.3

masks = glob.glob(os.path.join(MASKS_PATH, '*{}*.tif'.format(MASK_ALGORITHM)))

with open(IMAGES_DATAFRAME, 'w') as f:
    writer = csv.writer(f, delimiter=',')

    for mask in tqdm(masks):
        _, mask_name = os.path.split(mask)

        image_name = mask_name.replace('_{}_'.format(MASK_ALGORITHM), '_')
        writer.writerow([image_name, mask_name])

df = pd.read_csv(IMAGES_DATAFRAME, header=None, names=['images', 'masks'])
images_df = df[['images']]
masks_df = df[['masks']]

x_train, x_test, y_train, y_test = train_test_split(images_df, masks_df,
                                                    test_size=1 - TRAIN_RATIO, random_state=RANDOM_STATE)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=TEST_RATIO/(TEST_RATIO + VALIDATION_RATIO),
                                                random_state=RANDOM_STATE)

x_train.to_csv(os.path.join(OUTPUT, 'images_train.csv'), index=False)
y_train.to_csv(os.path.join(OUTPUT, 'masks_train.csv'), index=False)
x_val.to_csv(os.path.join(OUTPUT, 'images_val.csv'), index=False)
y_val.to_csv(os.path.join(OUTPUT, 'masks_val.csv'), index=False)
x_test.to_csv(os.path.join(OUTPUT, 'images_test.csv'), index=False)
y_test.to_csv(os.path.join(OUTPUT, 'masks_test.csv'), index=False)
