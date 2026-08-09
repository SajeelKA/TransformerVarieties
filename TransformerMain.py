import os
import requests
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass, asdict

import AttentionTypes
import FeedForwardTypes
import KvCacheTypes
import PositionalEmbeddingTypes
import Training
from datetime import datetime


def saveModel(model, config, pathReq = None):

    fileName = 'model_' + str(datetime.now()).replace('-','').replace(':','').replace(' ','')[:14] + '.pth'
    
    if pathReq is None:
      filePath = os.path.join(os.getcwd(), 'SavedModels', fileName) 
    else:
      filePath =  os.path.join(pathReq, fileName) 

    torch.save({"state_dict": model.state_dict(),"config": asdict(config)}, filePath)

    print('model saved in {}'.format(filePath))

def loadModel(pathReq):

    ckpt = torch.load(pathReq)    
    cfg = Training.Configs(**ckpt["config"])

    model = Transformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    return model
   
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = self.pick_attention_type(config)
        
        if config.use_experts==True:
            self.feed_forward = FeedForwardTypes.MixtureOfExperts(config, self.pick_feed_forward_type)
        else:
            self.feed_forward = self.pick_feed_forward_type(config) 
            

        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
    def pick_feed_forward_type(self, config):
        feed_forward_types = FeedForwardTypes.FeedForwardCatalog(config)
        if self.config.ff == 'mlp_with_relu':
            mlp = feed_forward_types.mlp_with_relu()
        elif self.config.ff == 'mlp_with_gelu':
            mlp = feed_forward_types.mlp_with_gelu()
        if self.config.ff == 'mlp_with_swiglu':
            mlp = feed_forward_types.mlp_with_swiglu()
        return mlp

    def pick_attention_type(self, config):
        if self.config.attention_type == 'multi_head_attention':
            return AttentionTypes.MultiHeadAttention(config)
        elif self.config.attention_type == 'sliding_window_attention':
            return AttentionTypes.SlidingWindowAttention(config)
        elif self.config.attention_type == 'multi_query_attention':
            return AttentionTypes.MultiQueryAttention(config)
        elif self.config.attention_type == 'grouped_query_attention':
            return AttentionTypes.GroupedQueryAttention(config)
        
        
    def forward(self, x, pos_embed, offset = 0):                
        a, other_outputs = self.attention(self.ln1(x), pos_embed, offset)
        x = x + a
        x = x + self.feed_forward(self.ln2(x))

        return x, other_outputs  
        
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_embedding = PositionalEmbeddingTypes.PositionEmbeddingCatalog(config)
        self.register_buffer('pos_freq_to_embed', self.pos_embedding.sinusoidal().to(device=config.device)) #buffer so it moves with model.to(device)
        self.pos = nn.Embedding(config.vocab_size, config.d_model)        
        self.tok = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(config) for i in range(config.num_blocks)])
        self.ln_final = nn.LayerNorm(config.d_model)
        self.out_logits = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x, offset = 0):        #offset needed for kv cache optimized generation
        if self.config.pos_embed == 'sinusoidal':
            x = self.pos_freq_to_embed[offset:offset+x.shape[1]] + self.tok(x) #did up to x.shape[0] because generation might need different batch size                
        elif self.config.pos_embed == 'rope':
            x = self.tok(x) #did up to x.shape[0] because generation might need different batch size                
        for block in self.blocks:            
            x, _ = block(x, self.pos_embedding, offset)             #offset for kvcache purposes            
        x = self.out_logits(self.ln_final(x))
       
        return x, _
        
        

