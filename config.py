from typing import TypedDict

class Config(TypedDict):
    n_layers:int 
    d_model:int 
    eps:float
    hidden_size_multiplier:int 
    num_heads:int
    context_len:int 
    dropout:float
    qkv_bias:bool 
    vocab_size:int
    

