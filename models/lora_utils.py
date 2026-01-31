# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

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
    ):
        super(LoRALayer, self).__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank, dtype=torch.float32))
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.randn(rank, out_dim) * std_dev )
        # self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        
        if norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = lambda x: x
         
    
    def forward(self, x):
        x = self.alpha * (x @ self.A) @ self.B 
        return self.norm(x)

class SimpleAdapter(nn.Module):
    def __init__(
        self,
        in_dim=256,
        out_dim=256,
        adapter_dim=32,
    ):
        super().__init__()
        self.down_proj = nn.Linear(in_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, out_dim)
        self.non_linearity = nn.ReLU()
        
    def forward(self, x):
        res = x
        x = self.down_proj(x)
        x = self.non_linearity(x)
        x = self.up_proj(x)
        return x + res

class MultiLoRALayer(nn.Module):
    def __init__(
        self,
        in_dim, 
        out_dim,
        rank=32,
        alpha=1.0,
        lora_num=1,
        norm=False,
    ):
        super(MultiLoRALayer, self).__init__()
        
        self.lora_layers = _get_clones(LoRALayer(in_dim, out_dim, rank, alpha, norm=norm), lora_num)
        # lora_layers = [LoRALayer(in_dim, out_dim, rank, alpha, norm=norm) for i in range(lora_num)]
        # self.lora_layers = nn.ModuleList(lora_layers)
        
    def forward(self, x, lora_index=0):
        return self.lora_layers[lora_index](x)
    