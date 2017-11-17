import torch
import numpy as np


class ToLabel(object):
    def __call__(self, image):
        """
        This class helps convert label PIL.Image to label torch.LongTensor.
        Args:
            image: PIL.Image, size (h, w), label data
        Returns:
                 : Torch.LongTensor, size (h, w), label data
        """
        return torch.from_numpy(np.array(image)).long()


class Relabel(object):
    """
    This class helps convert old lable to new lable. For example, in voc2012,
    we convert void label 255 to background label 0.
    """
    def __init__(self, olabel, nlabel):
        """
        Args:
            olabel: int, old label
            nlabel: int, new label
        """
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor
 
 
class DownSampleLabel(object):
    """
    This class helps downsample the label to the desired size.
    """
    def __init__(self, size):
        """
        Args:
            size: int or (int, int), the desired size (height, width)
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = (size[1], size[0])

    def __call__(self, label):
        """
        Args:
            label: PIL.Image
        Returns:
                 : the resized PIL.Image
        """
        return label.resize(self.size, Image.NEAREST)
