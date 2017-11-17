import torch
import torch.nn as nn
import torch.nn.init as init

import numpy as np


class UNet(nn.Module):
    def __init__(self, num_classes=21):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        # left_conv1 - left_conv5 are the same as VGG16 except the using of max pool
        # contracting paths
        # the left part of 'U'-Net
        self.left_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.left_conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.left_conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.left_conv4= nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        # pool -> conv -> up-conv 
        self.middle = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        )
        # expansive paths
        # the right part of 'U'-Net
        self.right_conv4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        )
        self.right_conv3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        )
        self.right_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )
        self.right_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        # get the results
        self.score = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)
    
    def forward(self, input_):
        # input_ size 16k x 16k
        # conventional conv step
        # le_conv1 size 16k x 16k
        le_conv1 = self.left_conv1(input_)
        # le_conv2 size 8k x 8k
        le_conv2 = self.left_conv2(le_conv1)
        # le_conv3 size 4k x 4k
        le_conv3 = self.left_conv3(le_conv2)
        # le_conv4 size 2k x 2k
        le_conv4 = self.left_conv4(le_conv3)
        
        # output size 2k x 2k -> 1k x 1k -> 2k x 2k
        output = self.middle(le_conv4)
        concat = torch.cat((le_conv4, output), dim=1)
        # output size 4k x 4k
        output = self.right_conv4(concat)
        concat = torch.cat((le_conv3, output), dim=1)
        # output size 8k x 8k
        output = self.right_conv3(concat)
        concat = torch.cat((le_conv2, output), dim=1)
        # output size 16k x 16k
        output = self.right_conv2(concat)
        concat = torch.cat((le_conv1, output), dim=1)
        # output size 16k x 16k (right_conv1() no upsample)
        output = self.right_conv1(concat)
        
        output = self.score(output)
        return output
    
    def initialize(self, state_dict):
        """
        Initialize some of the parameters(left_conv1 - left_conv5, parts of middle) by
        VGG16 pretrained on ImageNet
        Args:
            state_dict: torch.load('vgg16-397923af.pth')
        """
        maps = {
            'left_conv1.0.weight': 'features.0.weight',
            'left_conv1.0.bias': 'features.0.bias',
            'left_conv1.2.weight': 'features.2.weight',
            'left_conv1.2.bias': 'features.2.bias',
            'left_conv2.1.weight': 'features.5.weight',
            'left_conv2.1.bias': 'features.5.bias',
            'left_conv2.3.weight': 'features.7.weight',
            'left_conv2.3.bias': 'features.7.bias',
            'left_conv3.1.weight': 'features.10.weight',
            'left_conv3.1.bias': 'features.10.bias',
            'left_conv3.3.weight': 'features.12.weight',
            'left_conv3.3.bias': 'features.12.bias',
            'left_conv3.5.weight': 'features.14.weight',
            'left_conv3.5.bias': 'features.14.bias',
            'left_conv4.1.weight': 'features.17.weight',
            'left_conv4.1.bias': 'features.17.bias',
            'left_conv4.3.weight': 'features.19.weight',
            'left_conv4.3.bias': 'features.19.bias',
            'left_conv4.5.weight': 'features.21.weight',
            'left_conv4.5.bias': 'features.21.bias',
            'middle.1.weight': 'features.24.weight',
            'middle.1.bias': 'features.24.bias',
            'middle.3.weight': 'features.26.weight',
            'middle.3.bias': 'features.26.bias',
            'middle.5.weight': 'features.28.weight',
            'middle.5.bias': 'features.28.bias'
        }

        own_state = self.state_dict()
        for name in own_state.keys():
            if name in maps:
                try:
                    own_state[name].copy_(state_dict[maps[name]])
                except Exception:
                    raise RuntimeError('dimensions not match!')
            elif 'bias' in name:
                    init.constant(own_state[name], val=0)
            else:
                init.xavier_normal(own_state[name])
