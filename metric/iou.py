# Originally written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np

__all__ = ["IOU"]


class IOU(object):
    def __init__(self, num_classes, label_to_name=None):
        self.num_classes = num_classes

        default_label_to_name = {i: 'class %d' % i for i in range(num_classes)}
        if label_to_name is None:
            label_to_name = default_label_to_name
        self.label_to_name = label_to_name

        self.hist = np.zeros((num_classes, num_classes))
        self.iou = np.zeros(num_classes + 1)

    def reset(self):
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.iou = np.zeros(self.num_classes + 1)

    def update(self, labels_pred, labels_true):
        """
        Update the histogram
        Args:
            labels_pred:
            labels_true: both integer Tensor with size [batch_size, height, width]
        """
        assert labels_true.size() == labels_pred.size(), 'dimensions not match!'
        assert len(labels_true.size()) == 3, 'both should be 3D tensor'
        labels_true = labels_true.cpu().numpy()
        labels_pred = labels_pred.cpu().numpy()

        for lt, lp in zip(labels_true, labels_pred):
            self.hist += self.fast_hist(lt.flatten(), lp.flatten())

    def fast_hist(self, label_true, label_pred):
        x = label_true >= 0
        y = label_true < self.num_classes
        mask = np.array([x[i] and y[i] for i in range(len(y))])

        hist = np.bincount(
            self.num_classes * label_true[mask].astype(np.int) + label_pred[mask],
            minlength=self.num_classes ** 2
        )
        return hist.reshape(self.num_classes, self.num_classes)

    def print_iou(self):
        self.iou[0:self.num_classes] = np.diag(self.hist) / \
                                       (self.hist.sum(axis=0) + self.hist.sum(axis=1) - np.diag(self.hist))
        self.iou[self.num_classes] = self.iou.sum() / self.num_classes

        for i in range(self.num_classes):
            print('{}: {:.4f}'.format(self.label_to_name[i], self.iou[i]))
        print('mean IOU over {:d} classes: {:.4f}'.format(self.num_classes, self.iou[self.num_classes]))
