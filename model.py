import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
from IPython.display import display, Image
import tiktoken
from torch.utils.data import DataLoader, Dataset
from config import Config


# Layer Normalization layer
class LayerNormalization(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config.eps
        self.scale = nn.Parameter(torch.ones((config.d_model)))
        self.shift = nn.Parameter(torch.zeros((config.d_model))) 
        
    def forward(self, x: torch.Tensor):
        x_mean = x.mean(dim=-1, keepdim=True) 
        x_std = x.std(dim=-1, keepdim=True)
        x_norm = (x - x_mean) / (x_std + self.eps)
        return x * self.scale + self.shift
    
    

# FeedForward Layer
class FeedForwardLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config.d_model
        hidden_size_multiplier = config.hidden_size_multiplier
        
        self.ff_layer = nn.Sequential(
            nn.Linear(d_model, hidden_size_multiplier),
            nn.GELU(),
            nn.Linear(hidden_size_multiplier, d_model)        
        )
        
    def forward(self, x):
        return x + self.ff_layer(x)
    

class SelfAttentionLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.query_weights = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.key_weights = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.value_weights = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.num_heads = config.num_heads
        assert config.d_model % config.num_heads == 0, "d_model should be divisible by num_heads"
        self.h_dmodel = config.d_model // config.num_heads
        self.neg_inf = - 1e+5
        self.drop_out = torch.nn.Dropout(config.dropout)
        self.register_buffer('casual_mask', tensor=torch.triu(torch.ones(config.context_len, config.context_len), diagonal=1).bool())
    
    def forward(self, x):
        # x: (B, S, d_model)
        qeury_vectors = self.query_weights(x)
        key_vectors = self.key_weights(x)
        value_vectors = self.value_weights(x)
        batch_size, seq_len, d_model = x.size()
        
        # (B,S,d_model) -> (B, S, num_head, h_dmodel)
        qeury_vectors = qeury_vectors.view(batch_size, seq_len, self.num_heads, self.h_dmodel)
        key_vectors = key_vectors.view(batch_size, seq_len, self.num_heads, self.h_dmodel)
        value_vectors = value_vectors.view(batch_size, seq_len, self.num_heads, self.h_dmodel)
        
        # (B, Seq, num_heads, h_dmodel) -> (B, num_heads, Seq, h_dmodel)
        qeury_vectors = torch.permute(qeury_vectors, dims=(0, 2, 1, 3))
        key_vectors = torch.permute(key_vectors, dims=(0, 2, 1, 3))
        value_vectors = torch.permute(value_vectors, dims=(0, 2, 1, 3))
        mask = self.casual_mask[ :seq_len, : seq_len]
        
        # mask = self.casual_mask[:seq_len, :seq_len]  # (S, S)
        # mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)
         
        attention_scores = self.calculate_attention_score(qeury_vectors, key_vectors, mask)
        contextualized_vectores = attention_scores @ value_vectors
        
        # (B, num_heads, seq, head_d) => (B, seq, num_head, head_d) => (b, seq, d_model)
        contextualized_vectores = torch.permute(contextualized_vectores, dims=(0, 2, 1, 3))
        contextualized_vectores = contextualized_vectores.contiguous().view(batch_size, seq_len, self.num_heads*self.h_dmodel)
        contextualized_vectores = self.out_proj(contextualized_vectores)
        return (contextualized_vectores, attention_scores)
    
    def calculate_attention_score(self, qeury, key, mask):
        # (B,NumHeads,Seq, h_dmodel) * (B,num_heads,h_model, seq) => (B,num_heads, seq, seq)
        k_dmodel = key.size(-1)
        attention_scores = (qeury @ key.transpose(-1,-2)) / math.sqrt(k_dmodel)
        attention_scores = torch.masked_fill(attention_scores, mask=mask, value=self.neg_inf)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        return self.drop_out(attention_scores)
    
class TransformerBlock(nn.Module):
    def __init__(self,config: Config):
        super().__init__()
        self.attention = SelfAttentionLayer(config)
        self.layer_norm1 = LayerNormalization(config)
        self.layer_norm2 = LayerNormalization(config)
        self.feedforward = FeedForwardLayer(config)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        shortcut = x # Residual connection
        x = self.layer_norm1(x)
        x, _ = self.attention(x)
        x = self.dropout(x)
        x = shortcut + x
        
        shortcut = x
        x = self.layer_norm2(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        x = x + shortcut
        return x
    
    
