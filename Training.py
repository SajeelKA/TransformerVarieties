import os
import requests
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass
import TransformerMain
import DataLoader
import argparse

att_dict = {'mha' : 'multi_head_attention', 
            'window': 'sliding_window_attention', 
            'mqa': 'multi_query_attention',
            'gqa': 'grouped_query_attention',
            'flash': 'flash_attention',
           'paged': 'paged_attention'}

ff_dict = {'relu' : 'mlp_with_relu', 'gelu': 'mlp_with_gelu', 'swiglu': 'mlp_with_swiglu'}
lr_decay = {'cosine': 'cosine', 'linear':'linear'}

@dataclass
class Configs:
    device: str = 'cpu' if not torch.cuda.is_available() else 'cuda'
    vocab_size: int = 65
    seq_length: int = 512    
    max_seq_length: int = 1024
    batch_size: int = 32
    d_model: int = 64
    num_heads: int = 8
    num_blocks: int = 4
    num_groups: int = 4 
    attention_type: str = att_dict['gqa']
    ff: str = ff_dict['relu']
    lr_warmup: bool = True
    lr_decay: str = lr_decay['cosine']
    gradient_clipping: bool = True
    inference: bool = False
    window_size: int = 128
    epochs: int = 100
    pos_embed: str = 'rope' #could be rope or sinusoidal

config = Configs()

torch.set_num_threads(4)

def training():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", required = False, type=int, default = 64)
    parser.add_argument("--attention_type", required = False, default = 'gqa')
    parser.add_argument("--ff", required = False, default = 'relu')
    parser.add_argument("--num_blocks", required = False, type=int, default = 4)
    parser.add_argument("--num_groups", required = False, type=int, default = 4)
    parser.add_argument("--pos_embed", required = False, default = 'rope')
    parser.add_argument("--lr", required = False, type=float, default = 1e-3)
    parser.add_argument("--epochs", required = False, type=int, default = 100)
    
        
    args = parser.parse_args()
    
    config.d_model = args.d_model
    config.attention_type = att_dict[args.attention_type]
    config.ff = ff_dict[args.ff]
    config.num_blocks = args.num_blocks
    config.num_groups = args.num_groups
    config.pos_embed = args.pos_embed 
    config.lr = args.lr
    config.epochs = args.epochs
        
    data, gTokens, gVocabSize = DataLoader.getRawData()
    vocab = gTokens # !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
    #make dict where value is index of token and key is token itself
    tokenMapping = {k:v for v, k in enumerate(vocab)} 
    #changing character to index-based integer
    numericRepresentation = DataLoader.wordToTensor(data,tokenMapping)
    
    t = TransformerMain.Transformer(config).to(device=config.device)
    print('\n architecture:\n', t)
    n_params = sum(p.numel() for p in t.parameters())
    print('num parameters = ', n_params)
    optimizer = torch.optim.Adam(t.parameters(), lr = config.lr)
    num_batches = len(data)//config.batch_size
    print_after_fraction = 1000
    i = 0

    for e in range(args.epochs):
        for batchData, forwardPrediction in DataLoader.tinyDataLoader(numericRepresentation, config.batch_size, config.seq_length):
            x = torch.tensor(batchData).to(device=config.device)        
            logits, _ = t(x) #[B, S, V] raw logits - no argmax, cross_entropy needs the distribution to backprop through
            targets = torch.tensor(forwardPrediction).long().to(device=config.device) #[B, S] class indices
            optimizer.zero_grad()
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1)) #flatten to [B*S, V] vs [B*S]
            loss.backward()
            optimizer.step()
            i+=1

            if i % (num_batches//print_after_fraction)==0:
                print('epoch', e, 'batch number', i, 'loss', loss.item())

    TransformerMain.saveModel(t, t.config) 
 

if __name__ == '__main__':
    training()
