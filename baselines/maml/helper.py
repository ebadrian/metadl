""" Neural networks definition could be defined here and imported in model.py.
This file example is just meant to let you know you can create other python
scripts than model.py to organize your code.

"""
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,\
     BatchNormalization
from tensorflow.keras.initializers import GlorotUniform

tf.random.set_seed(1234)
def conv_net(nbr_classes, img_size = 28):
     """Reproduces the CNN used in the MAML paper. It was originally designed in
     Vinyals and al. (2016) .
     Conv layers kernels are initialized with Glorot Uniform by default.

     Args:
          nbr_classes: Integer, the number of classes.
          img_size: Integer, the width and height of the squarred images.
     """
     model = Sequential()
     model.add(Conv2D(64, (3, 3), strides = (2, 2), activation='relu',
          input_shape=(img_size, img_size, 3),
          kernel_initializer=GlorotUniform(seed=1234)))
     model.add(BatchNormalization())

     model.add(Conv2D(64, (3, 3), strides = (2, 2), activation='relu',
          kernel_initializer=GlorotUniform(seed=1234)))
     model.add(BatchNormalization())

     model.add(Conv2D(64, (3, 3), strides = (2, 2), activation='relu',
          kernel_initializer=GlorotUniform(seed=1234)))
     model.add(BatchNormalization())

     model.add(Flatten())
     model.add(Dense(nbr_classes, activation = 'softmax',
          kernel_initializer=GlorotUniform(seed=1234))) # Outputting probas
     return model

