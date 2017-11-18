import torch.nn as nn
import torch.nn.init as init


class DeepLab7x7(nn.Module):
    """
        input size: (3, 306, 306)
        output size: (num_classes, 38, 38)

        The network architecture is described in paper
        Semantic Image Segmentation With Deep Convolution Nets And Fully Connected CRFs
        https://arxiv.org/abs/1412.7062
    """

    def __init__(self, num_classes):
        super(DeepLab7x7, self).__init__()
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            # conv2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            # conv3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            # conv4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            # conv5
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.score = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=12, dilation=4),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, num_classes, kernel_size=1, stride=1)
        )

    def forward(self, input_):
        output = self.features(input_)
        output = self.score(output)
        return output

    def initialize(self, state_dict):
        maps = {
            'score.0.weight': 'classifier.0.weight',
            'score.0.bias': 'classifier.0.bias',
            'score.3.weight': 'classifier.3.weight',
            'score.3.bias': 'classifier.3.bias',
        }
        own_state = self.state_dict()
        for name in own_state.keys():
            if name in state_dict:
                try:
                    own_state[name].copy_(state_dict[name])
                except Exception:
                    raise RuntimeError('dimensions not match!')
            elif name in maps:
                if name == 'score.0.weight':
                    param = state_dict[maps[name]].view(4096, 512, 7, 7)
                elif name == 'score.3.weight':
                    param = state_dict[maps[name]].view(4096, 4096, 1, 1)
                else:
                    # bias
                    param = state_dict[maps[name]]
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('dimensions not match!')
        init.xavier_normal(own_state['score.6.weight'])
        init.constant(own_state['score.6.bias'], val=0)


class DeepLabLargeFOV(nn.Module):
    """
        input size: (3, 321, 321)
        output size: (num_classes, 41, 41)

        The network architecture is described in paper
        Semantic Image Segmentation With Deep Convolution Nets And Fully Connected CRFs
        https://arxiv.org/abs/1412.7062
        Caffe model can be found at http://liangchiehchen.com/projects/DeepLab-LargeFOV.html
    """

    def __init__(self, num_classes):
        super(DeepLabLargeFOV, self).__init__()
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # conv5
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.score = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=12, dilation=12),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1)
        )

    def forward(self, input_):
        output = self.features(input_)
        output = self.score(output)
        return output

    def initialize(self, state_dict):
        own_state = self.state_dict()
        for name in own_state.keys():
            if name in state_dict:
                try:
                    own_state[name].copy_(state_dict[name])
                except Exception:
                    raise RuntimeError('dimensions not match!')
            else:
                if 'weight' in name:
                    init.xavier_normal(own_state[name])
                if 'bias' in name:
                    init.constant(own_state[name], val=0)


class DeepLabMScLargeFOV(nn.Module):
    """
        input size: (3, 321, 321)
        output size: (num_classes, 41, 41)

        The network architecture is described in paper
        Semantic Image Segmentation With Deep Convolution Nets And Fully Connected CRFs
        https://arxiv.org/abs/1412.7062
        Caffe model can be found at http://liangchiehchen.com/projects/DeepLab-MSc-LargeFOV.html
    """

    def __init__(self, num_classes):
        super(DeepLabMScLargeFOV, self).__init__()
        self.num_classes = num_classes
        # multi scale predictions
        # first layer: 128 3x3 convolution filters
        # second layer: 128 1x1 convolution filters
        self.input_to_score = self.direct_link(in_channels=3, stride=8)
        self.pool1_to_score = self.direct_link(in_channels=64, stride=4)
        self.pool2_to_score = self.direct_link(in_channels=128, stride=2)
        self.pool3_to_score = self.direct_link(in_channels=256, stride=1)
        self.pool4_to_score = self.direct_link(in_channels=512, stride=1)
        # network architecture
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.score = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=12, dilation=12),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1)
        )

    def forward(self, input_):
        # forward process
        pool1 = self.layer1(input_)
        pool2 = self.layer2(pool1)
        pool3 = self.layer3(pool2)
        pool4 = self.layer4(pool3)
        pool5 = self.layer5(pool4)
        score = self.score(pool5)
        # multi scale predictions
        score0 = self.input_to_score(input_)
        score1 = self.pool1_to_score(pool1)
        score2 = self.pool2_to_score(pool2)
        score3 = self.pool3_to_score(pool3)
        score4 = self.pool4_to_score(pool4)
        # fuse
        output = score0 + score1 + score2 + score3 + score4 + score
        return output

    def initialize(self, state_dict):
        maps = {
            'layer1.0.weight': 'features.0.weight',
            'layer1.0.bias': 'features.0.bias',
            'layer1.2.weight': 'features.2.weight',
            'layer1.2.bias': 'features.2.bias',
            'layer2.0.weight': 'features.5.weight',
            'layer2.0.bias': 'features.5.bias',
            'layer2.2.weight': 'features.7.weight',
            'layer2.2.bias': 'features.7.bias',
            'layer3.0.weight': 'features.10.weight',
            'layer3.0.bias': 'features.10.bias',
            'layer3.2.weight': 'features.12.weight',
            'layer3.2.bias': 'features.12.bias',
            'layer3.4.weight': 'features.14.weight',
            'layer3.4.bias': 'features.14.bias',
            'layer4.0.weight': 'features.17.weight',
            'layer4.0.bias': 'features.17.bias',
            'layer4.2.weight': 'features.19.weight',
            'layer4.2.bias': 'features.19.bias',
            'layer4.4.weight': 'features.21.weight',
            'layer4.4.bias': 'features.21.bias',
            'layer5.0.weight': 'features.24.weight',
            'layer5.0.bias': 'features.24.bias',
            'layer5.2.weight': 'features.26.weight',
            'layer5.2.bias': 'features.26.bias',
            'layer5.4.weight': 'features.28.weight',
            'layer5.4.bias': 'features.28.bias'
        }
        own_state = self.state_dict()
        for name in own_state.keys():
            if name in maps:
                try:
                    own_state[name].copy_(state_dict[maps[name]])
                except Exception:
                    raise RuntimeError('dimensions not match')
            else:
                if 'weight' in name:
                    init.xavier_normal(own_state[name])
                if 'bias' in name:
                    init.constant(own_state[name], val=0)

    def direct_link(self, in_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(128, 128, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1)
        )
