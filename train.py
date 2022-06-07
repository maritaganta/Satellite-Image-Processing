import os
import pandas as pd

#  TODO: Plot history?

MASK_ALGORITHM = 'Kumar-Roy'

N_FILTERS = 64
N_CHANNELS = 3

EPOCHS = 50
BATCH_SIZE = 16
IMAGE_SIZE = (256, 256)
MODEL_NAME = 'unet'

IMAGES_PATH = '../../FireDetectionProcesses/dataset/images/patches'
MASKS_PATH = '../../FireDetectionProcesses/dataset/masks/patches'

IMAGES_DATAFRAME = '../../FireDetectionProcesses/dataset/images_masks.csv'

OUTPUT_DIR = '../../FireDetectionProcesses/dataset/train_output'

WORKERS = 4

CHECKPOINT_PERIOD = 5
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}.hdf5'.format(MODEL_NAME, MASK_ALGORITHM)

INITIAL_EPOCH = 0
RESTART_FROM_CHECKPOINT = None
if INITIAL_EPOCH > 0:
    RESTART_FROM_CHECKPOINT = os.path.join(OUTPUT_DIR, 'checkpoint-{}-{}-epoch_{:02d}.hdf5'.format(MODEL_NAME,
                                                                                                   MASK_ALGORITHM,
                                                                                                   INITIAL_EPOCH))

FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_final_weights.h5'.format(MODEL_NAME, MASK_ALGORITHM)

CUDA_DEVICE = 1

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

df = pd.read_csv(IMAGES_DATAFRAME, header=None, names=['images', 'masks'])

images_df = df[['images']]
masks_df = df[['masks']]

x_train = pd.read_csv('./dataset/images_train.csv')
y_train = pd.read_csv('./dataset/masks_train.csv')
x_val = pd.read_csv('./dataset/images_val.csv')
y_val = pd.read_csv('./dataset/masks_val.csv')
x_test = pd.read_csv('./dataset/images_test.csv')
y_test = pd.read_csv('./dataset/masks_test.csv')

images_train = [os.path.join(IMAGES_PATH, image) for image in x_train['images']]
masks_train = [os.path.join(MASKS_PATH, mask) for mask in y_train['masks']]

images_validation = [os.path.join(IMAGES_PATH, image) for image in x_val['images']]
masks_validation = [os.path.join(MASKS_PATH, mask) for mask in y_val['masks']]

train_generator = generator_from_lists(images_train, masks_train, batch_size=BATCH_SIZE)
validation_generator = generator_from_lists(images_validation, masks_validation, batch_size=BATCH_SIZE)