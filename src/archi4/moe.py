import torch.nn as nn
import torch
from torch import Tensor
from typing import List

class Moe(nn.Module):
    def __init__(self,
                 num_src_sim: int,
                 emb_size: int,
                 nhead: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 temprature: float = 1):
        super(Moe, self).__init__()
        self.num_src_sim = num_src_sim

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(512 * self.num_src_sim, 2048)
        self.linear2 = nn.Linear(2048, self.num_src_sim)

        self.softmax = nn.Softmax(dim=1)
        self.softmax_temprature = temprature


    def forward(self, input: Tensor,):
        x = self.transformer_encoder(input)                         # 122*96*512 → 122*96*512
        x = self.pool(x.transpose(0,2)).squeeze().transpose(0,1)    # 122*96*512 → 96*512

        x = torch.reshape(x, (-1, 512 * self.num_src_sim))          # 32*1536

        x = self.linear1(x)                                         # →32*2048
        x = self.linear2(x)                                         # →32*3

        return self.softmax(x/self.softmax_temprature)

