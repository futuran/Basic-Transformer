import torch.nn.functional as F
import torch
from torch import nn
from torch import Tensor


class OrigCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super(OrigCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor, batch_weight: Tensor) -> Tensor:
        L = input.shape[0]  # 51
        N = input.shape[1]  # 160
        C = input.shape[2]  # 37982

        input = F.log_softmax(input, dim=2)
        ignore_matrix = target != self.ignore_index
        target = F.one_hot(target, num_classes=C)
        loss_matrix = torch.sum(input*target, dim=2)

        loss_matrix *= ignore_matrix

        loss = torch.sum(loss_matrix)

        return - loss / torch.sum(ignore_matrix)


class SentWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super(SentWeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor, batch_weight: Tensor) -> Tensor:
        L = input.shape[0]  # 51
        N = input.shape[1]  # 160
        C = input.shape[2]  # 37982

        input = F.log_softmax(input, dim=2)
        ignore_matrix = target != self.ignore_index
        target = F.one_hot(target, num_classes=C)
        loss_matrix = torch.sum(input*target, dim=2)

        loss_matrix *= ignore_matrix
        loss = torch.sum(torch.mv(loss_matrix, batch_weight))

        return - loss / torch.sum(ignore_matrix)
