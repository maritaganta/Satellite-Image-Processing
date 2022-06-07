import pandas as pd
from generator import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from models import *
from plot_history import plot_history

PLOT_HISTORY = True
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

EARLY_STOP_PATIENCE = 5

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

model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1],
                  n_filters=N_FILTERS, n_channels=N_CHANNELS)

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=EARLY_STOP_PATIENCE)
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME),
                             monitor='loss', verbose=1, save_best_only=True, mode='auto', period=CHECKPOINT_PERIOD)

if INITIAL_EPOCH > 0:
    model.load_weights(RESTART_FROM_CHECKPOINT)

print('Training using {}...'.format(MASK_ALGORITHM))
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(images_train) // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(images_validation) // BATCH_SIZE,
    callbacks=[checkpoint, es],
    epochs=EPOCHS,
    workers=WORKERS,
    initial_epoch=INITIAL_EPOCH
)
print('Train finished!')


print('Saving weights')
model_weights_output = os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT)
model.save_weights(model_weights_output)
print("Weights Saved: {}".format(model_weights_output))

if PLOT_HISTORY:
    plot_history(history, OUTPUT_DIR)
