'''
    The code is based on TS2Vec: Towards Universal Representation of Time Series by Zhihan Yue et al., published in AAAI. The GitHub repository for the code is: https://github.com/zhihanyue/ts2vec
'''

import torch
from torch import nn
import torch.nn.functional as F
import  numpy as np

def hierarchical_contrastive_loss(z1, z2, alpha=0.3, temporal_unit=0):  
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1: 
        if alpha != 0:  
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit: 
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2) 
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2) 
    if z1.size(1) == 1: 
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d 

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1: 
        return z1.new_tensor(0.) 
    z = torch.cat([z1, z2], dim=0)
    z = z.transpose(0, 1) 
    sim = torch.matmul(z, z.transpose(1, 2))  
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]  
    logits = -F.log_softmax(logits, dim=-1) 
    
    i = torch.arange(B, device=z1.device)

    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2

    return loss

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1) # B: batch_size T
    if T == 1: 
        return z1.new_tensor(0.) 
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C 

    sim = torch.matmul(z, z.transpose(1, 2))  
    # sim = cosine_similarity(sim, z) # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]   # B x 2T x (2T-1)
    logits = -F.log_softmax(logits, dim=-1) # B x 2T x (2T-1)
    
    t = torch.arange(T, device=z1.device)
    # logits[:, t, T + t - 1]： B * T
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


