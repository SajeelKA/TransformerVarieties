import os
import requests
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass
import DataLoader
import argparse
import TransformerMain
import time

def generateGreedy(model, prompt, tokenMapping, config, max_gen = 512):
    
    x =  torch.tensor(DataLoader.wordToTensor(prompt,tokenMapping))[None].to(device=config.device)
    model = model.eval()
    with torch.no_grad():
        for i in range(max_gen - x.shape[1]):            
            logits, _ = model(x[:, -config.seq_length:])            
            next_token_index = torch.argmax(logits, dim = -1)[:,[-1]]   # greedy sampling             
            out = torch.cat([x, next_token_index], dim = -1)
            x = out

    return ''.join(DataLoader.tensorToWord(out.squeeze(0), tokenMapping))

def generateGreedyWithKV(model, prompt, tokenMapping, config, max_gen = 512):
    x =  torch.tensor(DataLoader.wordToTensor(prompt,tokenMapping))[None].to(device=config.device)
    model = model.eval()
    
    with torch.no_grad():        
        logits, past = model(x)    
        out = torch.cat([x, logits[:,[-1],:].argmax(dim = -1)], dim = 1)
        
        for i in range(max_gen-x.shape[1]):              
            logits, _ = model(out[:,[-1]], i+x.shape[1]) #only need to calculate one token's importance score             
            next_token_index = torch.argmax(logits, dim = -1)[:,[-1]]   # greedy sampling             
            out = torch.cat([out, next_token_index], dim = -1)
            

    return ''.join(DataLoader.tensorToWord(out.squeeze(0), tokenMapping))

def generateGreedyWithSlidingKV(model, prompt, tokenMapping, config, max_gen = 512):
    x =  torch.tensor(DataLoader.wordToTensor(prompt,tokenMapping))[None].to(device=config.device)
    model = model.eval()
    
    with torch.no_grad():        
        logits, idx = model(x)
        next_token_index = logits.argmax(dim = -1)[:,[-1]]
        out = torch.cat([x, next_token_index], dim = 1)
        
        for i in range(max_gen-x.shape[1]):              
            logits, idx = model(out[:,[-1]], i+x.shape[1]) #only need to calculate one token's importance score             
            next_token_index = torch.argmax(logits, dim = -1)[:,[-1]]   # greedy sampling (idx is supposed to come from kv cache            
            out = torch.cat([out, next_token_index], dim = -1)            

    return ''.join(DataLoader.tensorToWord(out.squeeze(0), tokenMapping))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required = True, type = str)
    parser.add_argument("--gen_length", required = False, type = int, default = 512)
    parser.add_argument("--file_path", required = False, type=str, default = 'model_gqa_RoPE.pth')
    args = parser.parse_args()
    
    data, gTokens, gVocabSize = DataLoader.getRawData()
    vocab = gTokens # !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
    #make dict where value is index of token and key is token itself
    tokenMapping = {k:v for v, k in enumerate(vocab)} 
    #changing character to index-based integer
    numericRepresentation = DataLoader.wordToTensor(data,tokenMapping)

    fileName = args.file_path
    pathReq = os.path.join(os.getcwd(), 'SavedModels', fileName) 
    t = TransformerMain.loadModel(pathReq)
    t = t.to(device = t.config.device)

    #with absolute pos embedding will break down after config.seq_length, so capping at config.seq_length
    #so need to use RoPE if we want that
    t.config.inference = True 
    prompt = args.prompt

    start_time = time.perf_counter()
    #out_with_kv = generateGreedyWithSlidingKV(t, prompt, tokenMapping, t.config, args.gen_length)
    out_with_kv = generateGreedyWithKV(t, prompt, tokenMapping, t.config, args.gen_length)
    print('\nOutput:\n\n', out_with_kv)
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.6f} seconds")
    
"""
    print('\nWith KV Cache:\n\n', out_with_kv)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.6f} seconds")

    start_time = time.perf_counter()
    t.config.inference = False
    out_without_kv = generateGreedy(t, prompt, tokenMapping, t.config, args.gen_length+1)
    print('\n\nWithout KV Cache: \n\n', out_without_kv)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.6f} seconds")

    print('='*100)
    # will be equal only till training sequence length if sliding window cache isn't used
    print('\nAre both equal?',out_without_kv == out_with_kv) 
"""
    
   
