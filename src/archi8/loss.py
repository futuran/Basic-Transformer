from torch.nn.modules.loss import CrossEntropyLoss 
import torch.nn.functional as F
import torch
from torch import nn
from torch import Tensor
from typing import Optional

# class SentWeightedCrossEntropyLoss(CrossEntropyLoss):
#     def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
#                  reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
#         super(SentWeightedCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
#         self.ignore_index = ignore_index
#         self.label_smoothing = label_smoothing

#     def forward(self, input: Tensor, target: Tensor) -> Tensor:
#         return F.cross_entropy(input, target, weight=self.weight,
#                                ignore_index=self.ignore_index, reduction=self.reduction,
#                                label_smoothing=self.label_smoothing)

class SentWeightedCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0, device='cpu') -> None:
        super(SentWeightedCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction, device)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.device = device

    def forward(self, input: Tensor, target: Tensor, batch_weight: Tensor) -> Tensor:
        loss = torch.tensor(0, dtype=torch.float32).to(self.device)
        batch_size = batch_weight.shape[-1]
        for i in range(batch_size):
            loss += F.cross_entropy(input[:,i,:], target[:,i], weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing) * batch_weight[i]
        return loss / batch_size

