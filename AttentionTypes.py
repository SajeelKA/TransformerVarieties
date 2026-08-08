import os
import requests
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass
import KvCacheTypes

class BaseFeatures(nn.Module): 
    def __init__(self, config):        
        super().__init__()
        self.qkv = nn.Linear(config.d_model, config.d_model * 3)        
        self.merge_heads = nn.Linear(config.d_model, config.d_model)
        self.kv_cache = KvCacheTypes.KvCache(config) 
        self.sliding_cache = KvCacheTypes.KvSliding(config)
        self.num_heads = config.num_heads

class MultiHeadAttentionWithOutKV(nn.Module):
    def __init__(self, config): 
        super().__init__()
        self.config = config
        self.base_features = BaseFeatures(config)  
        self.register_buffer('blocked_mask',~torch.tril(torch.ones(config.max_seq_length, config.max_seq_length,  dtype=torch.bool)), persistent=False)
        self.scale = 1.0 / ((config.d_model // config.num_heads) ** 0.5)

    def forward(self, x):
        B, S, C = x.shape
        H = self.config.num_heads
        q, k, v = self.base_features.qkv(x).split(C, dim = -1)
        
        q = q.view(B, S, H, C//H).transpose(1,2) #B,H,S,head_dim
        k = k.view(B, S, H, C//H).transpose(1,2)
        v = v.view(B, S, H, C//H).transpose(1,2)
        
        # msk_indices = torch.tril(torch.ones(S, S, device=x.device))        
        # relations = torch.masked_fill((q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(C//H, dtype=torch.int, device=x.device)), msk_indices == 0, float("-inf"))

        blocked = self.blocked_mask[offset : offset+S, :k.shape[2]]
        relations = torch.masked_fill((q @ k.transpose(-2,-1)) * self.scale, blocked,  float("-inf"))   
        
        importance_map = torch.softmax(relations, dim = -1) @ v
        
        return self.base_features.merge_heads(importance_map.transpose(1,2).reshape(B, S, C)), None
        
class MultiHeadAttention(nn.Module):
    def __init__(self, config): 
        super().__init__()
        self.config = config
        self.base_features = BaseFeatures(config) 
        self.register_buffer('blocked_mask',~torch.tril(torch.ones(config.max_seq_length, config.max_seq_length,  dtype=torch.bool)), persistent=False)
        self.scale = 1.0 / ((config.d_model // config.num_heads) ** 0.5)
     
    
    def forward(self, x, pos_embed, offset=0):
        B, S, C = x.shape
        H = self.config.num_heads
        q, k, v = self.base_features.qkv(x).split(C, dim = -1)

        q = q.view(B, S, H, C//H).transpose(1,2) #B,H,S,head_dim
        k = k.view(B, S, H, C//H).transpose(1,2)
        v = v.view(B, S, H, C//H).transpose(1,2)  
                
        if self.config.pos_embed == 'rope':
            q = pos_embed.RoPE(q, offset)
            k = pos_embed.RoPE(k, offset)
        
        if self.config.inference:                                      
            k, v = self.base_features.kv_cache.update(k, v, offset)            
        
        #mask the required tokens (for training will be S x S, while for inference, will be [1 x curr_token_number])
        blocked = self.blocked_mask[offset : offset+S, :k.shape[2]] 
        relations = torch.masked_fill((q @ k.transpose(-2,-1)) * self.scale, blocked,  float("-inf"))  # inference [1 x curr_token_number] scores, training [S x S] scores 
        
        importance_map = torch.softmax(relations, dim = -1) @ v
        
        return self.base_features.merge_heads(importance_map.transpose(1,2).reshape(B, S, C)), None 


class SlidingWindowAttention(nn.Module):
    def __init__(self,config):    
        super().__init__()
        self.config = config
        self.base_features = BaseFeatures(config)
        self.window_size = config.window_size
        
        r = torch.arange(self.config.seq_length).unsqueeze(1).to(device=config.device)
        c = torch.arange(self.config.seq_length).unsqueeze(0).to(device=config.device)
        #each column will be compared to all rows and result will  be S x k.shape[2]
        #colNo <= rowNo as well as rowNo has to be only <window_size> bigger than colNo
        self.window_mask = ~((c <= r) & ((r - c) < self.window_size))
        self.scale = 1.0 / ((config.d_model // config.num_heads) ** 0.5)
        

        
    def forward(self, x, pos_embed, offset=0):
        B, S, C = x.shape
        H = self.config.num_heads
        q, k, v = self.base_features.qkv(x).split(C, dim = -1)

        q = q.view(B, S, H, C//H).transpose(1,2) #B,H,S,head_dim
        k = k.view(B, S, H, C//H).transpose(1,2)
        v = v.view(B, S, H, C//H).transpose(1,2)  
        
        if self.config.pos_embed == 'rope':
            q = pos_embed.RoPE(q, offset)
            k = pos_embed.RoPE(k, offset)
            
        if self.config.inference:                                      
            k, v, idx = self.base_features.sliding_cache.update(k, v, offset) 
            q_pos   = torch.arange(offset, offset + S, device=x.device)[:, None]     # [S x 1]
            slot_pos = (offset + S) - 1 - (((offset + S) - 1 - torch.arange(self.window_size, device=x.device)) % self.window_size)   # [window_size]
            # during generation, we only need this mask in the prefill phase (S>1) till when buffer is still not full
            allowed = (slot_pos[None,:] <= q_pos) & ((q_pos - slot_pos[None,:]) < self.window_size) & (slot_pos[None,:] >= 0) 
            blocked = ~allowed                                                       # [S x W]
        else:
            blocked = self.window_mask[offset : offset+S, :k.shape[2]]
            idx = 0
            
        relations = torch.masked_fill((q @ k.transpose(-2,-1)) * self.scale, blocked,  float("-inf"))  
        
        importance_map = torch.softmax(relations, dim = -1) @ v
        
        return self.base_features.merge_heads(importance_map.transpose(1,2).reshape(B, S, C)), idx

class MultiQueryAttention(nn.Module):
    def __init__(self, config): 
        super().__init__()
        self.config = config
        self.base_features = BaseFeatures(config)
        self.d_kv = config.d_model//self.config.num_heads
        self.q_for_mqa = nn.Linear(config.d_model, config.d_model)        
        self.k_for_mqa = nn.Linear(config.d_model, self.d_kv)        
        self.v_for_mqa = nn.Linear(config.d_model, self.d_kv)        
        
        self.register_buffer('blocked_mask',~torch.tril(torch.ones(config.max_seq_length, config.max_seq_length,  dtype=torch.bool)), persistent=False)
        self.scale = 1.0 / ((config.d_model // config.num_heads) ** 0.5)       
    
    def forward(self, x, pos_embed, offset=0):
        B, S, C = x.shape
        H = self.config.num_heads
        q = self.q_for_mqa(x)
        k = self.k_for_mqa(x)
        v = self.v_for_mqa(x)

        q = q.view(B, S, H, C//H).transpose(1,2) #B,H,S,head_dim
        k = k.view(B, S, 1, C//H).transpose(1,2)
        v = v.view(B, S, 1, C//H).transpose(1,2) 

        if self.config.pos_embed == 'rope':
            q = pos_embed.RoPE(q, offset)
            k = pos_embed.RoPE(k, offset)
        
        if self.config.inference:                                      
            k, v = self.base_features.kv_cache.update(k, v, offset)            
        
        #mask the required tokens (for training will be S x S, while for inference, will be [1 x curr_token_number])
        blocked = self.blocked_mask[offset : offset+S, :k.shape[2]] 
        # don't need to repeat number of heads in MQA as shape will automatically be broadcast per head when matmul happens
        relations = torch.masked_fill((q @ k.transpose(-2,-1)) * self.scale, blocked,  float("-inf"))  # inference [1 x curr_token_number] scores, training [S x S] scores 
        
        importance_map = torch.softmax(relations, dim = -1) @ v
        
        return self.base_features.merge_heads(importance_map.transpose(1,2).reshape(B, S, C)), None 

class GroupedQueryAttention(nn.Module):
    def __init__(self, config): 
        super().__init__()
        self.config = config
        self.base_features = BaseFeatures(config)
        self.n_kv_heads = self.config.num_heads // self.config.num_groups
        self.d_kv = config.d_model // self.config.num_groups
        
        self.q_for_gqa = nn.Linear(config.d_model, config.d_model)        
        self.k_for_gqa = nn.Linear(config.d_model, self.d_kv)        
        self.v_for_gqa = nn.Linear(config.d_model, self.d_kv)        
        
        self.register_buffer('blocked_mask',~torch.tril(torch.ones(config.max_seq_length, config.max_seq_length,  dtype=torch.bool)), persistent=False)
        self.scale = 1.0 / ((config.d_model // config.num_heads) ** 0.5)
      
    
    def forward(self, x, pos_embed, offset=0):
        B, S, C = x.shape
        H = self.config.num_heads
        
        q = self.q_for_gqa(x)
        k = self.k_for_gqa(x)
        v = self.v_for_gqa(x)

        # q, k, v = self.base_features.qkv(x).split(C, dim = -1)
        
        q = q.view(B, S, H, C//H).transpose(1,2) #B,H,S,head_dim
        k = k.view(B, S, self.n_kv_heads, C//H).transpose(1,2)
        v = v.view(B, S, self.n_kv_heads, C//H).transpose(1,2)
        


        if self.config.pos_embed == 'rope':
            q = pos_embed.RoPE(q, offset)
            k = pos_embed.RoPE(k, offset)
        
        if self.config.inference:                                      
            k, v = self.base_features.kv_cache.update(k, v, offset)            
        
        k = torch.repeat_interleave(k, self.config.num_groups, dim = 1)
        v = torch.repeat_interleave(v, self.config.num_groups, dim = 1)
        #mask the required tokens (for training will be S x S, while for inference, will be [1 x curr_token_number])
        blocked = self.blocked_mask[offset : offset+S, :k.shape[2]] 
        
        relations = torch.masked_fill((q @ k.transpose(-2,-1)) * self.scale, blocked,  float("-inf"))  # inference [1 x curr_token_number] scores, training [S x S] scores 
        
        importance_map = torch.softmax(relations, dim = -1) @ v
        
        return self.base_features.merge_heads(importance_map.transpose(1,2).reshape(B, S, C)), None 
