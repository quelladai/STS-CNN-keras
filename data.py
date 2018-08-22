from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import random
import cv2
#from data_argmentation import *
from PIL import Image
import gdal

def generatedata(path, batchsize):

    data = glob.glob(path + "masked\\*.tif")
    dataset = []
    labelset = []
    data_1_set = []

    cnt = 0
    while 1:
        for imgname in data:
            midname = imgname[imgname.rindex("\\") + 1:]

            data_masked = gdal.Open(path + "Image\\" + midname)
            data_temporal = gdal.Open(path + "Image\\" + midname)
            label = gdal.Open(path + "Image\\" + midname)

            data = img_to_array(data_masked).astype('float32')
            data_1 = img_to_array(data_temporal).astype('float32')
            label = img_to_array(label).astype('float32')

            data /= 255
            data_1 /= 255
            label /= 255

            dataset.append(data)
            labelset.append(label)
            data_1_set.append(data_1)

            cnt += 1
            if cnt == batchsize:
                yield (np.array(dataset), np.array(labelset) ,np.array(data_1_set))
                cnt = 0
                dataset = []
                labelset = []
                data_1_set = []