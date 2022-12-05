import torch.nn as nn
import torch
from torch import Tensor
from typing import List

class Moe(nn.Module):
    def __init__(self,
                 num_sim: int,
                 tgt_vocab_size,
                 emb_size: int,
                 nhead: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 temprature: float = 1,):
        super(Moe, self).__init__()
        self.num_sim = num_sim

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=emb_size*self.num_sim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder2 = nn.TransformerEncoderLayer(d_model=emb_size*self.num_sim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder3 = nn.TransformerEncoderLayer(d_model=emb_size*self.num_sim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.linear1 = nn.Linear(512 * self.num_sim, 2048)
        self.linear2 = nn.Linear(2048, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, self.num_sim)

        self.generator = nn.Linear(512 * self.num_sim, tgt_vocab_size)
        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)
        self.softmax_temprature = temprature


    def forward(self, input: Tensor,):
        x = torch.reshape(input, (input.shape[0], -1, 512 * self.num_sim))  # 169*128*512→169*32*2048
        x = self.transformer_encoder(x)                                     # 169*32*2048 → 169*32*2048
        x = self.transformer_encoder2(x)                                    # 169*32*2048 → 169*32*2048
        x = self.transformer_encoder3(x)                                    # 169*32*2048 → 169*32*2048
        outs = self.generator(x)
        return outs


class Moe_old1(nn.Module):
    def __init__(self,
                 num_sim: int,
                 emb_size: int,
                 nhead: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 temprature: float = 1):
        super(Moe, self).__init__()
        self.num_sim = num_sim

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.linear1 = nn.Linear(512 * self.num_sim, 2048)
        self.linear2 = nn.Linear(2048, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, self.num_sim)
        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)
        self.softmax_temprature = temprature


    def forward(self, input: Tensor,):
        x = self.transformer_encoder(input)                         # 122*96*512 → 122*96*512
        x = self.pool(x.transpose(0,2)).squeeze().transpose(0,1)    # 122*96*512 → 96*512

        x = torch.reshape(x, (-1, 512 * self.num_sim))              # 96*512     → 32*1536

        x = self.relu(self.linear1(x))                              # → 32*2048
        x = self.relu(self.linear2(x)) 
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))

        return self.softmax(x/self.softmax_temprature)

class Moe_old2(nn.Module):
    def __init__(self,
                 num_sim: int,
                 emb_size: int,
                 nhead: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 temprature: float = 1):
        super(Moe, self).__init__()
        self.num_sim = num_sim

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=emb_size*self.num_sim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.linear1 = nn.Linear(512 * self.num_sim, 2048)
        self.linear2 = nn.Linear(2048, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, self.num_sim)
        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)
        self.softmax_temprature = temprature


    def forward(self, input: Tensor,):
        x = torch.reshape(input, (input.shape[0], -1, 512 * self.num_sim))  # 169*128*512→169*32*2048
        x = self.transformer_encoder(x)                                     # 169*32*2048 → 169*32*2048
        x = self.pool(x.transpose(0,2)).squeeze().transpose(0,1)            # 169*32*2048 → 32*2048

        # x = torch.reshape(x, (-1, 512 * self.num_sim))                      # 96*512     → 32*1536

        x = self.relu(self.linear1(x))                                      # 32*2048 → 32*2048
        x = self.relu(self.linear2(x)) 
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))

        return self.softmax(x/self.softmax_temprature)
