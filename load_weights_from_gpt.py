import numpy as np
import torch

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_embedding.weight = assign(gpt.pos_embedding.weight, params['wpe'])
    gpt.token_embedding.weight = assign(gpt.token_embedding.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.decoder_block[b].attention.query_weights.weight = assign(
            gpt.decoder_block[b].attention.query_weights.weight, q_w.T)
        gpt.decoder_block[b].attention.key_weights.weight = assign(
            gpt.decoder_block[b].attention.key_weights.weight, k_w.T)
        gpt.decoder_block[b].attention.value_weights.weight = assign(
            gpt.decoder_block[b].attention.value_weights.weight, v_w.T)
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.decoder_block[b].attention.query_weights.bias = assign(
            gpt.decoder_block[b].attention.query_weights.bias, q_b)
        gpt.decoder_block[b].attention.key_weights.bias = assign(
            gpt.decoder_block[b].attention.key_weights.bias, k_b)
        gpt.decoder_block[b].attention.value_weights.bias = assign(
            gpt.decoder_block[b].attention.value_weights.bias, v_b)
        gpt.decoder_block[b].attention.out_proj.weight = assign(
            gpt.decoder_block[b].attention.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.decoder_block[b].attention.out_proj.bias = assign(
            gpt.decoder_block[b].attention.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])
        gpt.decoder_block[b].feedforward.ff_layer[0].weight = assign(
            gpt.decoder_block[b].feedforward.ff_layer[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.decoder_block[b].feedforward.ff_layer[0].bias = assign(
            gpt.decoder_block[b].feedforward.ff_layer[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.decoder_block[b].feedforward.ff_layer[2].weight = assign(
            gpt.decoder_block[b].feedforward.ff_layer[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.decoder_block[b].feedforward.ff_layer[2].bias = assign(
            gpt.decoder_block[b].feedforward.ff_layer[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        gpt.decoder_block[b].layer_norm1.scale = assign(
            gpt.decoder_block[b].layer_norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.decoder_block[b].layer_norm1.shift = assign(
            gpt.decoder_block[b].layer_norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.decoder_block[b].layer_norm2.scale = assign(
            gpt.decoder_block[b].layer_norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.decoder_block[b].layer_norm2.shift = assign(
            gpt.decoder_block[b].layer_norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_layernorm.scale = assign(gpt.final_layernorm.scale, params["g"])
    gpt.final_layernorm.shift = assign(gpt.final_layernorm.shift, params["b"])
    gpt.final_linear.weight = assign(gpt.final_linear.weight, params["wte"])
