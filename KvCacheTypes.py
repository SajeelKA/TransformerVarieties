import os
import requests
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass

class KvCache():
    def __init__(self, config,n_kv_heads=None):
        H = (config.num_heads // config.num_groups) if config.attention_type == 'grouped_query_attention' else config.num_heads
        self.config = config
        self.k = torch.zeros(config.batch_size, H, config.max_seq_length, config.d_model//config.num_heads, device=config.device)
        self.v = torch.zeros(config.batch_size, H,config.max_seq_length, config.d_model//config.num_heads, device=config.device)
        
        
    def update(self, k_new, v_new, offset):

        next_token = offset + k_new.shape[2]        
        self.k[:, :, offset:next_token] = k_new #save current position and add next sequence to the full array
        self.v[:, :, offset:next_token] = v_new  

         
        
        return self.k[:k_new.shape[0], :, :next_token], self.v[:v_new.shape[0], :, :next_token]

class KvSliding():
    def __init__(self, config):
        self.config = config
        self.k = torch.zeros(config.batch_size, config.num_heads, config.window_size, config.d_model//config.num_heads, device=config.device)
        self.v = torch.zeros(config.batch_size, config.num_heads,config.window_size, config.d_model//config.num_heads, device=config.device)
        self.filled = 0
        
    def update(self, k_new, v_new, offset):
        
        idx = offset % self.config.window_size        
        next_token = idx + k_new.shape[2] #need to double check        
        self.k[:, :, idx:next_token] = k_new
        self.v[:, :, idx:next_token] = v_new
        self.filled = min(self.filled + k_new.shape[2], self.config.window_size)
        
        return self.k[:k_new.shape[0], :, :], self.v[:v_new.shape[0], :, :], idx
        
