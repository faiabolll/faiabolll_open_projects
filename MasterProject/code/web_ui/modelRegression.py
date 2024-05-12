from yolov5 import detect
import glob
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import os
import numpy as np
import yaml

architecture_config = [
    (7, 8, 2, 3),
    "M",
    (3, 24, 1, 1),
    "M",
    (1, 16, 1, 0),
    (3, 32, 1, 1),
    (1, 32, 1, 0),
    (3, 64, 1, 1),
    "M",
    [(1, 64, 1, 0), (3, 128, 1, 1), 2],
    (3, 128, 1, 1),
    # (3, 1024, 2, 1),
    # (3, 1024, 1, 1),
    (3, 128, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class ScaledHardsigmoid(nn.Module):
    def __init__(self, scale = 2):
        super().__init__()
        self.scale = scale

    def forward(self, input):
        res = self.scale * nn.Hardsigmoid()(input)
        return res.double()

class YoloRegression(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(YoloRegression, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x, image_meta):
        # x, image_meta = x
        x = self.darknet(x)
        x = torch.cat((
            torch.flatten(x, start_dim=1),
            torch.tensor(image_meta)
        ), 1)
        x = self.fcs(x)
        return x

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self):

        return nn.Sequential(
            nn.Flatten(),
            # 128 *28 *28 out image's shape from darknet + coords of rezec
            nn.Linear(128 *28 * 28 + 4, 155),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(155, 1),
            ScaledHardsigmoid()
        )