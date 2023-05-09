import os
import numpy as np
#,import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import warnings
warnings.filterwarnings("ignore")

trainDirectory = 'train/train'

# creating black image

imageHeight = 100
imageWidth = 100
thickness = 3
inputShape = (imageHeight, imageWidth, thickness)

imageDataGenerator = ImageDataGenerator(rescale=1./255,
                                vertical_flip=True,
                                horizontal_flip=True,
                                rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                zoom_range=0.1,
                                validation_split=0.2)
# code explaination

# imageDataGenerator is used for data preprocessing and image augmentation on training images.
# rescale=1./255 rescales pixel values to be between 0 and 1.
# vertical_flip=True performs random vertical flips on the images.
# horizontal_flip=True performs random horizontal flips.
# rotation_range=40 randomly rotates the image by up to 40 degrees.
# width_shift_range=0.2 and height_shift_range=0.2 randomly shift the width and height of the images by up to 20%.
# zoom_range=0.1 randomly zooms the images by up to 10%.

testDataGenerator = ImageDataGenerator(rescale=1./255)

# testDataGenerator is used for data preprocessing on test images, and only rescales the pixel values to be between 0 and 1 using rescale=1./255.

trainGenerator = imageDataGenerator.flow_from_directory(trainDirectory,
                                                 shuffle=True,
                                                 batch_size=32,
                                                 subset='training',
                                                 target_size=(100, 100))

validGenerator = imageDataGenerator.flow_from_directory(trainDirectory,
                                                 shuffle=True,
                                                 batch_size=16,
                                                 subset='validation',
                                                 target_size=(100, 100))

# code explaination

# flow_from_directory() is a method of ImageDataGenerator and DirectoryIterator classes in the Keras library for loading and augmenting image data from a directory during model training or prediction.
# it takes so many arguments like:

# directory: the path to the directory containing the images.
# target_size: the size to which the images should be resized.
# color_mode: the color mode of the images ('rgb' or 'grayscale').
# batch_size: the number of images in each batch.
# shuffle: whether to shuffle the order of the images.
# class_mode: the type of label to use ('categorical', 'binary', 'sparse', or None).
# subset: whether to load a subset of the images ('training' or 'validation').

