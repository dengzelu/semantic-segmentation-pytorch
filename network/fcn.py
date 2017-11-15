import torch
import torch.nn as nn
import torch.nn.init as init

import numpy as np


class FCN32(nn.Module):
    def __init__(self, num_classes=21, upscore_ks=64, upscore_stride=32):
        super(FCN32, self).__init__()
        self.num_classes = num_classes
        self.upscore_ks = upscore_ks

        self.features = nn.Sequential(
            # conv 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv 5
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.score = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, num_classes, kernel_size=1, stride=1)
        )
        self.upscore = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=upscore_ks, stride=upscore_stride, bias=False
        )

    def forward(self, input_):
        output = self.features(input_)
        output = self.score(output)
        output = self.upscore(output)
        return output

    def initialize(self, state_dict):
        maps = {
            'classifier.0.weight': 'score.0.weight',
            'classifier.0.bias': 'score.0.bias',
            'classifier.3.weight': 'score.3.weight',
            'classifier.3.bias': 'score.3.bias',
        }

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            if name in own_state:
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('Dimensions does not match!')
            elif name in maps:
                if name == 'classifier.0.weight':
                    param = param.view(4096, 512, 7, 7)
                if name == 'classifier.3.weight':
                    param = param.view(4096, 4096, 1, 1)
                try:
                    own_state[maps[name]].copy_(param)
                except Exception:
                    raise RuntimeError('Dimensions does not match!')
                
        init.xavier_normal(own_state['score.6.weight'])
        init.constant(own_state['score.6.bias'], val=0)
        own_state['upscore.weight'].copy_(self.upscore_filter())

    def upscore_filter(self):
        # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
        factor = (self.upscore_ks + 1) // 2
        if self.upscore_ks % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:self.upscore_ks, :self.upscore_ks]

        filter_2d = torch.Tensor(
            (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        )
        filter_4d = torch.zeros(self.num_classes, self.num_classes, self.upscore_ks, self.upscore_ks)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                filter_4d[i, j, :, :] = filter_2d
        return filter_4d
