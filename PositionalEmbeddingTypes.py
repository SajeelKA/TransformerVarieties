import os
import requests
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass

class PositionEmbeddingCatalog():
    def __init__(self, config, base = 10000):
        self.embed_type = None
        self.config = config
        if self.config.pos_embed == 'sinusoidal': #takes total embed dim because we add it to the start
            d_req = config.d_model            
        elif self.config.pos_embed == 'rope': #works only on head dimension since it's implemented with attention
            d_req = config.d_model//config.num_heads
        
        self.i = torch.arange(d_req//2, device=config.device).int()
        self.pos = torch.arange(config.max_seq_length, device=config.device).int()
        #create entire table and store
        self.angles = self.pos[:, None]/torch.pow(base, ((2*(self.i))/d_req)) # (pos / base ^ (2i/d))   #[S, d_model/2]
        self.sin = torch.sin(self.angles) 
        self.cos = torch.cos(self.angles) 

    # one test might be to add 4 positional frequencies instead of just 2
    def sinusoidal(self, base = 10000):
        to_embed = torch.zeros((self.config.max_seq_length, self.config.d_model), device=self.config.device)   
        to_embed[:, 2*self.i] = self.sin #sin(pos / base ^ (2i/d)) for even angles #[B, S, d_model/2]
        to_embed[:, 2*self.i + 1] = self.cos #sin(pos / base ^ (2i/d)) for odd angles

        return to_embed

    def RoPE(self, x, offset = 0):

        # x1 = x[torch.arange(0,x.shape[-1], 2)]
        # x2 = x[torch.arange(1,x.shape[-1], 2)]        
        #broadcasting to B,S,H,D//2 (same operation over all B, H)
        # x1_prime = x1 * self.cos[None, :, None, :] - x2 * self.sin[None, :, None, :] 
        # x2_prime = x1 * self.sin[None, :, None, :] + x2 * self.cos[None, :, None, :]
        # rotated = torch.zeros(x.shape)
        # rotated[:,:,:,2*self.i] = x1_prime        
        # rotated[:,:,:,2*self.i + 1] = x2_prime  

        S = x.shape[2]
        assert offset + S <= self.config.max_seq_length, f'{offset+S} exceeds max_seq_length {self.config.max_seq_length}'
        rotated = torch.zeros(x.shape).to(device = x.device)      
        cos = self.cos[offset:offset+S][None, None]
        sin = self.sin[offset:offset+S][None, None]
        
        #broadcasting to B,H,S,D//2 (same operation over all B, H)
        rotated[:,:,:,2*self.i] = x[:,:,:,2*self.i] * cos - x[:,:,:,2*self.i+1] * sin
        rotated[:,:,:,2*self.i + 1] = x[:,:,:,2*self.i] * sin + x[:,:,:,2*self.i+1] * cos

        return rotated

