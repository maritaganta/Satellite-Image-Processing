import rasterio as rio
import os
from patchify import patchify
import cv2
#  arr_st = rio.open(path).read(bands).transpose((1, 2, 0))

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

train_img_dir = '../../FireDetectionProcesses/data/256_patches/images/'
train_mask_dir = '../../FireDetectionProcesses/data/256_patches/masks/'

img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)

num_images = len(os.listdir(train_img_dir))
