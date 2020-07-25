## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Conv2d:
        nn.init.xavier_normal(m.weight)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # input is of shape 224 x 224
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),   # img size = 224 - 4 = 220
            nn.CELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # img size = 220 / 2 = 110
            #nn.BatchNorm2d(num_features=32),
            nn.Conv2d(32, 64, kernel_size=3),           # img size = 110 - 2 = 108
            nn.CELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # img size = 108 / 2 = 54
            #nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 64, kernel_size=3),          # img size = 54 - 2 = 52
            nn.CELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # img size = 52 / 2 = 26
        )

        output_size = 68 * 2
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(33856, 1024),
            nn.CELU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.CELU(inplace=True),
            nn.Linear(1024, output_size),
        )

        self.features.apply(init_weights)
        self.classifier.apply(init_weights)

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
