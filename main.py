import gc
import datetime
import os

import pandas as pd
import numpy as np

from copy import deepcopy
from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from IPython.display import clear_output

from model import STSCNN

BATCH_SIZE = 2

masked_DIR = r"D:\data-for-STS\train\masked\\"
temporal_DIR = r"D:\data-for-STS\train\ori\\"
label_DIR = r"D:\data-for-STS\train\ori\\"

val_masked_DIR = r"D:\data-for-STS\val\masked\\"
val_temporal_DIR = r"D:\data-for-STS\val\ori\\"
val_label_DIR = r"D:\data-for-STS\val\ori\\"

test_masked_DIR = r"D:\data-for-STS\test\masked\\"
test_temporal_DIR = r"D:\data-for-STS\test\ori\\"
test_label_DIR = r"D:\data-for-STS\test\ori\\"

class DataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
        while True:
            temporal = next(generator)

            data_gen_args = dict(rotation_range=20,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 rescale=1. / 255,
                                 horizontal_flip=True)
            datagen = ImageDataGenerator(**data_gen_args)
            masked_generator = datagen.flow_from_directory(
                masked_DIR,
                batch_size=BATCH_SIZE,
                target_size=(512, 512),
                class_mode=None,
                shuffle=True,
                seed=1)

            masked = next(masked_generator)

            gc.collect()
            yield [masked, temporal]


train_datagen = DataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    horizontal_flip=True
)
train_generator_1 = train_datagen.flow_from_directory(temporal_DIR,
                                                      target_size=(512, 512),
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=True,
                                                      seed=1
                                                      )

train_gen_args = dict(rotation_range=20,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     rescale=1. / 255,
                     horizontal_flip=True)
train_label_datagen = ImageDataGenerator(**train_gen_args)
train_generator_2 = train_label_datagen.flow_from_directory(label_DIR,
                                                            target_size=(512, 512),
                                                            batch_size=BATCH_SIZE,
                                                            class_mode=None,
                                                            shuffle=True,
                                                            seed=1)
train_generator = zip(train_generator_1, train_generator_2)

class DataGenerator_val(ImageDataGenerator):
    def flow_from_directory(self, directory, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
        while True:
            temporal = next(generator)

            data_gen_args = dict(rescale=1. / 255)
            datagen = ImageDataGenerator(**data_gen_args)
            masked_generator = datagen.flow_from_directory(
                val_masked_DIR,
                batch_size=BATCH_SIZE,
                target_size=(512, 512),
                class_mode=None,
                shuffle=True,
                seed=1)

            masked = next(masked_generator)

            gc.collect()
            yield [masked, temporal]

val_datagen = DataGenerator_val(rescale=1. / 255)
val_generator_1 = val_datagen.flow_from_directory(val_temporal_DIR,
                                                  target_size=(512, 512),
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True,
                                                  seed=1
                                                  )

val_gen_args = dict(rescale=1. / 255)
val_label_datagen = ImageDataGenerator(**val_gen_args)
val_generator_2 = val_label_datagen.flow_from_directory(val_label_DIR,
                                                        target_size=(512, 512),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode=None,
                                                        shuffle=True,
                                                        seed=1)
val_generator = zip(val_generator_1, val_generator_2)


class DataGenerator_test(ImageDataGenerator):
    def flow_from_directory(self, directory, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
        while True:
            temporal = next(generator)

            data_gen_args = dict(rescale=1. / 255)
            datagen = ImageDataGenerator(**data_gen_args)
            masked_generator = datagen.flow_from_directory(
                test_masked_DIR,
                batch_size=BATCH_SIZE,
                target_size=(512, 512),
                class_mode=None,
                shuffle=True,
                seed=1)

            masked = next(masked_generator)

            gc.collect()
            yield [masked, temporal]


test_datagen = DataGenerator_test(rescale=1. / 255)
test_generator_1 = test_datagen.flow_from_directory(test_temporal_DIR,
                                                    target_size=(512, 512),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    seed=1
                                                    )

test_gen_args = dict(rescale=1. / 255)
test_label_datagen = ImageDataGenerator(**test_gen_args)
test_generator_2 = test_label_datagen.flow_from_directory(test_label_DIR,
                                                          target_size=(512, 512),
                                                          batch_size=BATCH_SIZE,
                                                          class_mode=None,
                                                          shuffle=True,
                                                          seed=1)
test_generator = zip(test_generator_1, test_generator_2)

test_1 = next(test_generator_1)
test_2 = next(test_generator_2)
(masked, temporal) = test_1
ori = test_2

def plot_callback(model):
    """Called at the end of each epoch, displaying our previous test images,
    as well as their masked predictions and saving them to disk"""

    # Get samples & Display them
    pred_img = model.predict([masked, temporal])
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Clear current output and display test images
    for i in range(len(ori)):
        _, axes = plt.subplots(1, 4, figsize=(50, 5))
        axes[0].imshow(masked[i, :, :, :])
        axes[1].imshow(pred_img[i, :, :, :] * 1.)
        axes[2].imshow(ori[i, :, :, :])
        axes[3].imshow(temporal[i, :, :, :])
        axes[0].set_title('Masked Image')
        axes[1].set_title('Predicted Image')
        axes[2].set_title('Original Image')
        axes[3].set_title('alluxiary Image')

        plt.savefig(r'D:\STSCNN\test_samples\test\img_{}_{}.png'.format(i, pred_time))
        plt.close()

model = STSCNN(weight_filepath='D:/STSCNN/logs/test/')

model.fit(
    train_generator,
    steps_per_epoch=1000,
    validation_data=val_generator,
    validation_steps=100,
    epochs=500,
    plot_callback=plot_callback,
    callbacks=[
        TensorBoard(log_dir='D:/STSCNN/logs/new_training', write_graph=False)
    ]
)
