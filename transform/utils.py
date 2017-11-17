import torch
import numpy as np


class ToLabel(object):
    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long()


class Relabel(object):
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor
 
 
class DownSampleLabel(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = (size[1], size[0])

    def __call__(self, label):
        return label.resize(self.size, Image.NEAREST)
