# STS-CNN-keras

The Keras implementation of "Missing Data Reconstruction in Remote Sensing Image With a Unified Spatial–Temporal–Spectral Deep Convolutional Neural Network", https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8316243.

It is just an unofficial pytorch implementation of the paper, i just finish it but didn't test the result.

Python 3.6
Keras 2.2.0
Tensorflow 1.8

The file 'model.py' is the model in the paper.

Use the original image as the label and the temporal information, add the mask to the original images to generate the masked images, After two epoch, the result is as follow:


