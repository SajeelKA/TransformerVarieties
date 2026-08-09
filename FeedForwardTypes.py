import os
import requests
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass

class FeedForwardCatalog(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config        

    def mlp_with_relu(self):
        model = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_model * 4),
                nn.ReLU(),
                nn.Linear(self.config.d_model * 4, self.config.d_model)
            )

        return model

    def mlp_with_gelu(self):
        model = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_model * 4),
                nn.GELU(),
                nn.Linear(self.config.d_model * 4, self.config.d_model)
            )

        return model

    def mlp_with_swiglu(self):
        model = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_model * 4),
                nn.GELU(), # have to add swiglu functionality
                nn.Linear(self.config.d_model * 4, self.config.d_model)
            )

        return model

class MixtureOfExperts(nn.Module):
    def __init__(self, config, mlp):
        super().__init__()
        self.config = config
        self.ff = self.config.ff
        self.experts  = nn.ModuleList([mlp(self.config) for i in range(self.config.num_experts)])
        self.router = nn.Linear(config.d_model, config.num_experts)

    def forward(self, x):
        B, S, C = x.shape
        xf = x.reshape(B*S, C)

        #get scores on which token will be routed to which expert
        logits = self.router(xf) # [B, E] 
        # choose highest 2 expert scores
        top_k_logits, top_ks = logits.topk(self.config.top_k_experts, dim = -1) # [B, k] 
        # how much importance to give to each top-k expert
        gate_weights = torch.softmax(top_k_logits, dim = -1) 
        
        out = torch.zeros_like(xf) 
        
        # loop through each expert to get which ones will be activated according to top k scores
        for i in range(self.config.num_experts):
            # get all tokens to be routed to expert[i], and which of their k slots expert[i] sits in        
            idx, k = (top_ks == i).nonzero(as_tuple=True) # idx is token number, k is index k of the top k expert            
            
            if idx.numel() == 0: #in case no tokens are going to this expert
                continue          
            
            # run expert e on ONLY its tokens, weight by that token's gate for this  slot
            out[idx] += gate_weights[idx, k, None] * self.experts[i](xf[idx]) #sum(g{i}E{i}); i = top(k), g=softmax(E[i]) = softmax(E[i]) * E[i]

        # output shape is same as what a regular FFN shape is
        return out.view(B, S, C)
