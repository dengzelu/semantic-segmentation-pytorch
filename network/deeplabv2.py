# Originally written by isht7
# https://github.com/isht7/pytorch-deeplab-resnet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DeepLabMScResNet101"]


class Bottleneck(nn.Module):
    def __init__(self, in_channels, channels, stride=1, dilation=1, shortcut=None):
        super(Bottleneck, self).__init__()
        # strided conv1 is for downsample
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, affine=True)
        # size + 2 * padding - [2 * (dilation - 1) + 3] + 1 =  size
        # so padding = dilation
        padding = dilation
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                               padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, affine=True)
        self.conv3 = nn.Conv2d(channels, channels * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 4, affine=True)

        self.relu = nn.ReLU(inplace=True)
        self.shortcut = shortcut

    def forward(self, input_):
        residual = input_

        output = self.conv1(input_)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)
        if self.shortcut is not None:
            residual = self.shortcut(input_)
        output += residual
        output = self.relu(output)

        return output


class ASPPClassifier(nn.Module):
    def __init__(self, dilation_series, padding_series, num_classes):
        super(ASPPClassifier, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, num_classes, kernel_size=3, stride=1,
                                              padding=padding, dilation=dilation, bias=True))

    def forward(self, input_):
        output = self.conv2d_list[0](input_)
        for i in range(1, len(self.conv2d_list)):
            output += self.conv2d_list[i](input_)
        return output


class DeepLabResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(DeepLabResNet, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.last_channels = 64
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.aspp_pred = self._make_pred_layer(ASPPClassifier, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

    def _make_layer(self, block, channels, num_blocks, stride, dilation):
        # inchannels是输入该层的channels
        layer_shortcut = None
        layer_shortcut = nn.Sequential(
            nn.Conv2d(self.last_channels, channels * 4, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(channels * 4, affine=True)
        )
        layers = []
        layers.append(block(self.last_channels, channels, stride, dilation, shortcut=layer_shortcut))
        self.last_channels = channels * 4
        for _ in range(num_blocks - 1):
            layers.append(block(self.last_channels, channels, dilation=dilation))
        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward(self, input_):
        output = self.conv(input_)
        output = self.bn(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.aspp_pred(output)
        return output


class DeepLabMScResNet101(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabMScResNet101, self).__init__()
        self.network = DeepLabResNet(Bottleneck, (3, 4, 23, 3), num_classes)

    def forward(self, input_):
        in_size = input_.size()[2]
        inputs = []
        inputs.append(input_)
        size = (int(in_size * 0.75) + 1, int(in_size * 0.75) + 1)
        inputs.append(F.upsample(input_, size, mode='bilinear'))
        size = (int(in_size * 0.5) + 1, int(in_size * 0.5) + 1)
        inputs.append(F.upsample(input_, size, mode='bilinear'))

        outputs = [self.network(data) for data in inputs]
        size = (self.out_size(in_size), self.out_size(in_size))
        outputs[1] = F.upsample(outputs[1], size, mode='bilinear')

        temp = torch.max(outputs[0], outputs[1])
        outputs.append(torch.max(temp, F.upsample(outputs[2], size, mode='bilinear')))
        return outputs

    @staticmethod
    def out_size(in_size):
        # conv: kernel_size=7, stride=2, padding=3
        out_size = (in_size + 1) // 2
        # maxpool: kernel_size=3, stride=2, padding=1, ceil_mode=True
        out_size = int(math.ceil((out_size + 1) / 2))
        # layer2 first conv: kernel_size=1, stride=2, padding=0
        out_size = (out_size + 1) // 2
        # all the three dowmsample ops in this network
        return out_size
