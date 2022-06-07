import threading
import random
import rasterio
import os
import numpy as np
import sys
from sklearn.utils import shuffle as shuffle_lists

BANDS: tuple = (8, 3, 2)
MAX_PIXEL_VALUE = 2**16 - 1


class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


def get_image(path):
    img = rasterio.open(path).read(BANDS).transpose((1, 2, 0))
    img = np.float32(img)/MAX_PIXEL_VALUE
    return img


def get_mask(path):
    mask = rasterio.open(path).read().transpose((1, 2, 0))
    seg = np.float32(mask)
    return seg


@threadsafe_generator
def generator_from_lists(images_path, masks_path, batch_size=16, shuffle=True):
    images = []
    masks = []
    while True:
        if shuffle:
            images_path, masks_path = shuffle_lists(images_path, masks_path)

        for img_path, mask_path in zip(images_path, masks_path):
            img = get_image(img_path)
            mask = get_mask(mask_path)
            images.append(img)
            masks.append(mask)

        if len(images) >= batch_size:
            yield np.array(images), np.array(masks)
            images = []
            masks = []
