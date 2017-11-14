import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy2d(nn.Module):
    def __init__(self, weight=None):
        super(CrossEntropy2d, self).__init__()
        self.loss = nn.NLLLoss(weight)

    def forward(self, logits, labels):
        return self.loss(F.log_softmax(logits), labels)
        
