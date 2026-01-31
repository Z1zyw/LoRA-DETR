
from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import Embedding, Linear
import torch.nn.functional as F
import copy
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class LoRALayer(nn.Module):
    def __init__(
        self,
        in_dim, 
        out_dim,
        rank=32,
        alpha=1.0,
        norm=False,
        lr_alpha=1,
    ):
        super(LoRALayer, self).__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank, dtype=torch.float32))
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        # self.B = nn.Parameter(torch.randn(rank, out_dim) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        if norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = lambda x: x
         
    
    def forward(self, x):
        x = self.alpha * (x @ self.A) @ self.B
        
        return self.norm(x)

class MultiLoRALayer(nn.Module):
    def __init__(
        self,
        in_dim, 
        out_dim,
        rank=32,
        alpha=1.0,
        lora_num=1,
        norm=False,
        xavier_init=False
    ):
        super(MultiLoRALayer, self).__init__()
        
        self.lora_layers = _get_clones(LoRALayer(in_dim, out_dim, rank, alpha, norm=norm, lr_alpha=1), lora_num)
            
        if xavier_init:
            for layer in self.lora_layers:
                nn.init.xavier_uniform_(layer.A)
                nn.init.xavier_normal_(layer.B)
        
    def forward(self, x, lora_index=0):
        return self.lora_layers[lora_index](x)
    
class ProgressiveMultiLoRALayer(nn.Module):
    def __init__(
        self,
        in_dim, 
        out_dim,
        rank=16,
        alpha=1.0,
        lora_num=1,
    ):
        super(ProgressiveMultiLoRALayer, self).__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank, dtype=torch.float32))
        self.A = nn.Parameter(torch.randn(in_dim, rank * lora_num) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank * lora_num, out_dim))
        # self.B = nn.Parameter(torch.randn(rank * lora_num, out_dim) * std_dev)
        self.alpha = alpha
        self.r = rank
    
    def forward(self, x, lora_idx=0):
        # A = self.A[:, lora_idx * self.r : (lora_idx + 1) * self.r]
        # B = self.B[lora_idx * self.r : (lora_idx + 1) * self.r]
        A = self.A[:, : (lora_idx + 1) * self.r]
        B = self.B[: (lora_idx + 1) * self.r]
        x = self.alpha * (x @ A @ B)
        return x

    