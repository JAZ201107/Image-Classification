import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNet(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.num_channels = params.num_channels

        # (B, C, H, W) -> (B, C', )
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)

        # (B, C, H', W') -> (B, C * 2 , H'', W'')
        self.conv2 = nn.Conv2d(
            self.num_channels, self.num_channels * 2, 3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(self.num_channels * 2)

        # (B, C, H, W) -> (B, C * 4, H, W)
        self.conv3 = nn.Conv2d(
            self.num_channels * 2, self.num_channels * 4, 3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(self.num_channels * 4)

        # fully connected layers to transform the output of the convolution layers to the final output

        #
        self.fc1 = nn.Linear(8 * 8 * self.num_channels * 4, self.num_channels * 4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels * 4)
        self.fc2 = nn.Linear(self.num_channels * 4, 6)
        self.dropout_rate = params.dropout_rate

    def forward(self, s):
        s = self.bn1(self.conv1(s))
        s = F.relu(F.max_pool2d(s, 2))
        s = self.bn2(self.conv2(s))
        s = F.relu(F.max_pool2d(s, 2))
        s = self.bn3(self.conv3(s))
        s = F.relu(F.max_pool2d(s, 2))

        s = s.view(-1, 8 * 8 * self.num_channels * 4)

        # Apply 2 fully connected layers
        s = F.dropout(
            F.relu(self.fcbn1(self.fc1(s))), p=self.dropout_rate, training=self.training
        )

        se = self.fc2(s)

        return F.log_softmax(s, dim=1)
