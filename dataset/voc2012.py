# some codes come from https://github.com/bodokaiser/piwise
import os
import math
import random

from PIL import Image
from torch.utils.data import Dataset

LABEL_TO_NAME = {
    0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow',
    11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
    16: 'potted plant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tv/monitor'
}


EXTENSIONS = ['.jpg', '.png']


def load_image(file):
    return Image.open(file)


def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)


def image_path(root, basename, extension):
    return os.path.join(root, '{basename}{extension}'.format(basename=basename, extension=extension))


def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


class VOC2012(Dataset):
    def __init__(self, root, input_transform=None, target_transform=None, size=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.size = size

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.size is not None:
            image, label = self.random_crop(image, label)

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)

    def random_crop(self, image, label):
        if isinstance(self.size, int):
            self.size = (self.size, self.size)

        nh, nw = self.size  # desired image height and width
        w, h = image.size  # current image height and width
        h_ratio = nh / h
        w_ratio = nw / w

        if h_ratio > 1 or w_ratio > 1:
            ratio = max(h_ratio, w_ratio)
            image = image.resize((math.ceil(w * ratio), math.ceil(h * ratio)), Image.BILINEAR)
            label = label.resize((math.ceil(w * ratio), math.ceil(h * ratio)), Image.NEAREST)
        w, h = image.size
        assert h >= nh and w > nw, 'method VOC2012.random_crop() fails'
        i, j = random.randint(0, h - nh), random.randint(0, w - nw)
        image = image.crop((j, i, j + nw, i + nh))
        label = label.crop((j, i, j + nw, i + nh))
        return image, label
