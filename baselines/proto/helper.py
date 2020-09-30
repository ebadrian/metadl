import gin
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,\
     BatchNormalization
from tensorflow.keras.initializers import GlorotUniform

tf.random.set_seed(1234)
def conv_net(nbr_classes, img_size = 28):
     """Reproduce the CNN used in the MAML paper. It was originally designed in
     Vinyals and al. (2016) .
     Conv layers kernels are initialized with Glorot Uniform by default."""

     model = Sequential()
     model.add(Conv2D(64, (3, 3), strides = (1, 1), activation='relu',\
          input_shape=(img_size, img_size, 3),
          kernel_initializer=GlorotUniform(seed=1234)))
     model.add(BatchNormalization())
     model.add(MaxPooling2D((2, 2)))

     model.add(Conv2D(64, (3, 3), strides = (1, 1), activation='relu',
          kernel_initializer=GlorotUniform(seed=1234)))
     model.add(BatchNormalization())
     model.add(MaxPooling2D((2, 2)))

     model.add(Conv2D(64, (3, 3), strides = (1, 1), activation='relu',
          kernel_initializer=GlorotUniform(seed=1234)))
     model.add(BatchNormalization())
     model.add(MaxPooling2D((2, 2)))

     model.add(Flatten())
     return model

