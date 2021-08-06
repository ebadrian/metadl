""" Neural networks definition could be defined here and imported in model.py.
This file example is just meant to let you know you can create other python
scripts than model.py to organize your code.

"""
import torch
from torch import nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)

#tf.random.set_seed(1234)
def conv_net(nbr_classes, img_size = 128):
     """
     Conv layers kernels are initialized with Glorot Uniform by default.
     Args:
          nbr_classes: Integer, the number of classes.
          img_size: Integer, the width and height of the squarred images.
     """
     net = nn.Sequential(
        nn.Conv2d(3, 128, 3),
        nn.BatchNorm2d(128, momentum=1, affine=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 64, 3),
        nn.BatchNorm2d(64, momentum=1, affine=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 3),
        nn.BatchNorm2d(64, momentum=1, affine=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(64, 64, 3),
        nn.BatchNorm2d(64, momentum=1, affine=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(64, 64, 3),
        nn.BatchNorm2d(64, momentum=1, affine=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),

        Flatten(),
        nn.Linear(256, nbr_classes))

     return net


