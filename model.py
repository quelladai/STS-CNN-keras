import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D,  Conv2DTranspose, Lambda
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import array_to_img, img_to_array
import keras as k
import glob
import data
import math
import cv2
from keras.models import Sequential

from keras.layers.convolutional import Conv2D, Deconv2D
from keras.layers import merge, Dropout, concatenate, add
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input
from keras.utils.np_utils import to_categorical
import random
import cv2 as cv
from keras.preprocessing.image import img_to_array, load_img
import keras
import gdal


class STSCNN(object):
    def __init__(self, img_rows = 512, img_cols = 512):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = self.STSNet()
        self.current_epoch = 0


    def STSNet(self):
        data = Input((self.img_rows, self.img_cols, 3))
        data_2 = Input((self.img_rows, self.img_cols, 3))

        #初始特征提取
        conv_1 = Conv2D(30, 3, activation='relu', strides=(1, 1), padding='same', kernel_initializer='he_normal')(data)
        conv_2 = Conv2D(30, 3, activation='relu', strides=(1, 1), padding='same', kernel_initializer='he_normal')(data_2)
        concat_1_2 = concatenate([conv_1, conv_2], axis=-1)

        #多尺度特征提取
        feature3 = Conv2D(20, 3,activation='relu',strides=(1, 1), padding='same', kernel_initializer='he_normal')(concat_1_2)
        feature5 = Conv2D(20, 5,activation='relu',strides=(1, 1), padding='same', kernel_initializer='he_normal')(concat_1_2)
        feature7 = Conv2D(20, 7,activation='relu',strides=(1, 1), padding='same', kernel_initializer='he_normal')(concat_1_2)
        concat_3_5_7 = concatenate([feature3, feature5, feature7], axis=-1)

        sum0 = add([concat_1_2,concat_3_5_7])

        conv1 = Conv2D(60, 3,activation='relu',strides=(1, 1), padding='same', kernel_initializer='he_normal')(sum0)
        conv2 = Conv2D(30, 3,activation='relu',strides=(1, 1), padding='same', kernel_initializer='he_normal')(conv1)
        sum1 = add([conv2, conv_2])

        #空洞卷积
        conv3 = Conv2D(60, 3, activation='relu', strides=(1, 1), dilation_rate=(2, 2), padding='same', kernel_initializer='he_normal')(sum1)
        conv4 = Conv2D(60, 3, activation='relu', strides=(1, 1), dilation_rate=(3, 3), padding='same', kernel_initializer='he_normal')(conv3)
        conv5 = Conv2D(60, 3, activation='relu', strides=(1, 1), dilation_rate=(2, 2), padding='same', kernel_initializer='he_normal')(conv4)
        sum2 = add([conv3, conv5])

        #普通卷积
        conv6 = Conv2D(60, 3, activation='relu', strides=(1, 1), padding='same', kernel_initializer='he_normal')(sum2)
        conv7 = Conv2D(3, 3, activation='relu', strides=(1, 1), padding='same', kernel_initializer='he_normal')(conv6)

        model = Model(inputs=[data, data_2], outputs=conv7)

        model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])

        return model

    def fit(self, generator, epochs=10, plot_callback=None, *args, **kwargs):
        """Fit the U-Net to a (images, targets) generator

        param generator: training generator yielding (maskes_image, original_image) tuples
        param epochs: number of epochs to train for
        param plot_callback: callback function taking Unet model as parameter
        """

        # Loop over epochs
        for _ in range(epochs):

            # Fit the model
            self.model.fit_generator(
                generator,
                epochs=self.current_epoch + 1,
                initial_epoch=self.current_epoch,
                *args, **kwargs
            )

            # Update epoch
            self.current_epoch += 1

            # After each epoch predict on test images & show them
            if plot_callback:
                plot_callback(self.model)

            # Save logfile
            if self.weight_filepath:
                self.save()