"""
MyModel model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #self.relu = nn.ReLU()
        #self.max = nn.MaxPool2d(kernel_size=2, stride=2)
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        conv_1_out_channels = 12
        conv_2_out_channels = 128
        conv_3_out_channels = 256
        self.stack = nn.Sequential(
            nn.BatchNorm2d(3),
            # 3 * 32 * 32
            nn.Conv2d(in_channels = 3,out_channels = conv_1_out_channels,kernel_size = 3, stride = 1, padding = 0), # 30 * 30
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 15 * 15
            nn.Conv2d(conv_1_out_channels, conv_2_out_channels, 3), #13 * 13
            nn.ReLU(),
            nn.Conv2d(conv_2_out_channels, conv_2_out_channels, 3), #11 * 11
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 5 * 5
            nn.Conv2d(conv_2_out_channels, conv_3_out_channels, 3), # 3 * 3
            nn.ReLU(),
            nn.Conv2d(conv_3_out_channels, conv_3_out_channels, 3), # 1 * 1
            nn.ReLU(),
            #nn.Conv2d(conv_2_out_channels, conv_3_out_channels, 5), 
            #nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4425984, 1024), # 4425984 is hardcoded to output of previous layers at the moment
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,10),
        )

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        #print('x', x.shape)
        logits = self.stack(x)
        return logits
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################