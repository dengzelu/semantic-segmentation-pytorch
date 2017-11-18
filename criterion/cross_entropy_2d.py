import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CrossEntropy2d"]


class CrossEntropy2d(nn.Module):
    def __init__(self, weight=None):
        super(CrossEntropy2d, self).__init__()
        self.loss = nn.NLLLoss(weight)

    def forward(self, logits, labels):
        return self.loss(F.log_softmax(logits), labels)
