import torch
import torch.nn as nn
import torch.nn.init as init

import numpy as np

"""
    training stage (for all model):
    input size: (32*k, 32*k), such as (320, 320), (480, 480)
    output size: the same as the input size
"""

__all__ = ["FCN32s", "FCN16s", "FCN8s"]

# Map the parameter name in our model to the public VGG16 parameter name
# It is used in method initialize(torch.load('/path/to/your/vgg16_00b39a1b.pth))
# VGG16 model https://github.com/jcjohnson/pytorch-vgg

GENERAL_MAPS = {
    'layer1.0.weight': 'features.0.weight', 'layer1.0.bias': 'features.0.bias',
    'layer1.2.weight': 'features.2.weight', 'layer1.2.bias': 'features.2.bias',
    'layer2.0.weight': 'features.5.weight', 'layer2.0.bias': 'features.5.bias',
    'layer2.2.weight': 'features.7.weight', 'layer2.2.bias': 'features.7.bias',
    'layer3.0.weight': 'features.10.weight', 'layer3.0.bias': 'features.10.bias',
    'layer3.2.weight': 'features.12.weight', 'layer3.2.bias': 'features.12.bias',
    'layer3.4.weight': 'features.14.weight', 'layer3.4.bias': 'features.14.bias',
    'layer4.0.weight': 'features.17.weight', 'layer4.0.bias': 'features.17.bias',
    'layer4.2.weight': 'features.19.weight', 'layer4.2.bias': 'features.19.bias',
    'layer4.4.weight': 'features.21.weight', 'layer4.4.bias': 'features.21.bias',
    'layer5.0.weight': 'features.24.weight', 'layer5.0.bias': 'features.24.bias',
    'layer5.2.weight': 'features.26.weight', 'layer5.2.bias': 'features.26.bias',
    'layer5.4.weight': 'features.28.weight', 'layer5.4.bias': 'features.28.bias',
    'layer6.0.weight': 'classifier.1.weight', 'layer6.0.bias': 'classifier.1.bias',
    'layer6.3.weight': 'classifier.4.weight', 'layer6.3.bias': 'classifier.4.bias'}


def upscore_filter(kernel_size, num_classes):
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
    # get the bilinear upsample 4D kernel
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]

    filter_2d = torch.from_numpy(
        (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    ).float()
    filter_4d = torch.zeros(num_classes, num_classes, kernel_size, kernel_size)
    for i in range(num_classes):
        filter_4d[i, i, :, :].copy_(filter_2d)
    return filter_4d


class _FCNBase(nn.Module):
    def __init__(self, num_classes):
        super(_FCNBase, self).__init__()
        # conv1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # conv2
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # conv3
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # conv4
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # conv5
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # fc to conv
        self.layer6 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, num_classes, kernel_size=1, stride=1)
        )

    def forward(self, *input_):
        pass

    def initialize(self, state_dict):
        """
            Initialize the model with state_dict

        """
        maps = GENERAL_MAPS
        own_state = self.state_dict()
        for name in own_state.keys():
            if name in maps:
                if name == 'layer6.0.weight':
                    param = state_dict[maps[name]].view(4096, 512, 7, 7)
                elif name == 'layer6.3.weight':
                    param = state_dict[maps[name]].view(4096, 4096, 1, 1)
                else:
                    param = state_dict[maps[name]]
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('dimensions not match!')
            else:
                if 'bias' in name:
                    init.constant(own_state[name], val=0)
                elif 'upscore' in name:
                    own_state[name].copy_(
                        upscore_filter(own_state[name].size(-1), self.num_classes)
                    )
                else:
                    init.xavier_normal(own_state[name])


class FCN32s(_FCNBase):
    def __init__(self, num_classes=21):
        super(FCN32s, self).__init__(num_classes)
        self.num_classes = num_classes
        # upsample 32x to 1x
        self.upscore = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=64, stride=32, padding=16, bias=False)

    def forward(self, input_):
        pool1 = self.layer1(input_)
        pool2 = self.layer2(pool1)
        pool3 = self.layer3(pool2)
        pool4 = self.layer4(pool3)
        pool5 = self.layer5(pool4)
        score_fc_32s = self.layer6(pool5)
        # upsample fc score map from 32x to 1x
        output = self.upscore(score_fc_32s)
        return output

    def initialize(self, state_dict):
        super(FCN32s, self).initialize(state_dict)


class FCN16s(_FCNBase):
    def __init__(self, num_classes=21):
        super(FCN16s, self).__init__(num_classes)
        self.num_classes = num_classes

        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1)
        # upsample 32x to 16x
        self.upscore_32s = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        # upsample 16x to 1x
        self.upscore_16s = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=32, stride=16, padding=8, bias=False)

    def forward(self, input_):
        pool1 = self.layer1(input_)
        pool2 = self.layer2(pool1)
        pool3 = self.layer3(pool2)
        pool4 = self.layer4(pool3)
        pool5 = self.layer5(pool4)
        score_pool4_16s = self.score_pool4(pool4)
        score_fc_32s = self.layer6(pool5)
        # upsample fc score map from 32x to 16x
        score_fc_16s = self.upscore_32s(score_fc_32s)
        # fuse pool4 score map(16x) and fc score map(16x)
        fuse_pool4_16s = score_fc_16s + score_pool4_16s
        # upsample fused score map from 16x to 1x
        output = self.upscore_16s(fuse_pool4_16s)
        return output

    def initialize(self, state_dict):
        super(FCN16s, self).initialize(state_dict)


class FCN8s(_FCNBase):
    def __init__(self, num_classes=21):
        super(FCN8s, self).__init__(num_classes)
        self.num_classes = num_classes

        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1)
        # upsample 32x to 16x
        self.upscore_32s = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        # upsample 16x to 8x
        self.upscore_16s = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        # upsample 8x to 1x
        self.upscore_8s = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False)

    def forward(self, input_):
        pool1 = self.layer1(input_)
        pool2 = self.layer2(pool1)
        pool3 = self.layer3(pool2)
        pool4 = self.layer4(pool3)
        pool5 = self.layer5(pool4)
        score_pool3_8s = self.score_pool3(pool3)
        score_pool4_16s = self.score_pool4(pool4)
        score_fc_32s = self.layer6(pool5)
        # upsample fc score map from 32x to 16x
        score_fc_16s = self.upscore_32s(score_fc_32s)
        # fuse pool4 score map(16x) and fc score map(16x)
        fuse_pool4_16s = score_fc_16s + score_pool4_16s
        # upsample fused score map from 16x to 8x
        score_fuse_8s = self.upscore_16s(fuse_pool4_16s)
        # fuse pool3 score map(8x) and fused score map(8x)
        fuse_pool3_8s = score_fuse_8s + score_pool3_8s
        # upsample fused score map from 8x to 1x
        output = self.upscore_8s(fuse_pool3_8s)
        return output

    def initialize(self, state_dict):
        super(FCN8s, self).initialize(state_dict)
