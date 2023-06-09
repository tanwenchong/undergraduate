import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import argparse
#from model import *
from lite import *
import torch.utils.data as Data
from einops.layers.torch import Rearrange
from einops import rearrange
from transformers import get_linear_schedule_with_warmup

def generate_square_subsequent_mask(sz: int, device='cpu'):
	return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

def get_key_padding_mask(tokens):
    key_padding_mask = torch.zeros(tokens.size())
    key_padding_mask[tokens == 0] = float('-inf')
    return key_padding_mask


class PositionalEncoding(nn.Module):
	def __init__(self, d_model: int, dropout: float = 0.5, max_len: int = 5000):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)
		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(max_len, 1, d_model)
		pe[:, 0, 0::2] = torch.sin(position * div_term)
		pe[:, 0, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)
	def forward(self, x) :
		x = x + self.pe[:x.size(0)].requires_grad_(False)
		return self.dropout(x)

class transformer_model(nn.Module):
	def __init__(self,d_model=256,nhead=16,tgt_size=25,BATCH=10,trg_mask=None):    
		super().__init__()
		position = PositionalEncoding(d_model)
		self.transformer=nn.Transformer(d_model)
		self.d_model=d_model
		self.BATCH=BATCH
		self.embed=nn.Sequential(nn.Embedding(tgt_size,self.d_model),position)
		self.projection = nn.Linear(d_model,tgt_size, bias=False)
		
	def forward(self,X,dec_inputs,trg_mask=None):
		#X=rearrange(enc_inputs,'a b c -> a (b c) ')
		src_key_padding_mask=get_key_padding_mask(X)
		X=self.embed(X)
		X=rearrange(X,'a b c -> b a c')
		y=self.embed(dec_inputs)
		y=rearrange(y,'a b c -> b a c')
		trg_mask=generate_square_subsequent_mask(sz=y.shape[0])
		output = self.transformer(X,y,tgt_mask=trg_mask,src_key_padding_mask=src_key_padding_mask)
		output=rearrange(output,'a b c -> b a c')
		return output

