@dataclass
class Config:
    n_layers:int = 12
    d_model:int = 768
    eps:float = 1e-5
    hidden_size_multiplier:int = 4 
    num_heads:int = 12
    context_len:int = 1024
    dropout:float = 0.1
    qkv_bias:bool = False
    vocab_size:int = 50257
    device:str = 'cuda'