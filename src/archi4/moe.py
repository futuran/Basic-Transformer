import torch.nn as nn
from torch import Tensor
from typing import List

class Moe(nn.Module):
    def __init__(self,
                 num_sim: int,
                 emb_size: int,
                 nhead: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Moe, self).__init__()

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(emb_size * num_sim, 2048)
        self.linear2 = nn.Linear(2048, num_sim)

    def forward(self, input: Tensor,):
        x = self.transformer_encoder(input)
        x = self.pool(x)
        x = x.flatten()
        x = self.linear1(x)
        x = self.linear2(x)

        return x

