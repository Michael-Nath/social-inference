from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights
import torch

from inference import ComputeGraphBuilder, ComputeGraph, ComputeGraphNode

config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# def llama_attention(b: ComputeGraphBuilder, layer_idx: int, head_dim: int, ):

def llama_rope(b: ComputeGraphBuilder, dim: int, q: ComputeGraphNode, k: ComputeGraphNode, cos: ComputeGraphNode, sin: ComputeGraphNode, unsqueeze_dim: int = 1):
    def rotate_half(x: ComputeGraphNode):
        x1 = b.slice("x1", x, 0, dim // 2)
        x2 = b.slice("x2", x, dim // 2, dim)
        # Create a fixed tensor of -1.0 with the same shape as x2
        neg_one = b.fixed("neg_one", torch.full((dim // 2,), -1.0))
        # Negate x2 by multiplying with -1.0 tensor
        neg_x2 = b.hadamard("neg_x2", x2, neg_one)
        return b.cat("rotated", neg_x2, x1, -1)
    
    # For single batch, q and k have shape [heads, seq_len, head_dim]
    # First unsqueeze cos and sin at the specified dimension
    cos_unsqueezed = b.unsqueeze("cos_unsqueezed", cos, unsqueeze_dim)
    sin_unsqueezed = b.unsqueeze("sin_unsqueezed", sin, unsqueeze_dim)
    
    # Then broadcast to match q and k dimensions
    cos_broadcast = b.broadcast("cos_broadcast_heads", cos_unsqueezed, 0, 1)  # Broadcast heads dim
    cos_broadcast = b.broadcast("cos_broadcast_seq", cos_broadcast, 1, 1)  # Broadcast seq dim
    
    sin_broadcast = b.broadcast("sin_broadcast_heads", sin_unsqueezed, 0, 1)  # Broadcast heads dim
    sin_broadcast = b.broadcast("sin_broadcast_seq", sin_broadcast, 1, 1)  # Broadcast seq dim
    
    # Apply RoPE to query
    q_rotated = rotate_half(q)
    q_cos = b.hadamard("q_cos", q, cos_broadcast)
    q_sin = b.hadamard("q_sin", q_rotated, sin_broadcast)
    q_embed = b.hadamard("q_embed", q_cos, q_sin)
    
    # Apply RoPE to key
    k_rotated = rotate_half(k)
    k_cos = b.hadamard("k_cos", k, cos_broadcast)
    k_sin = b.hadamard("k_sin", k_rotated, sin_broadcast)
    k_embed = b.hadamard("k_embed", k_cos, k_sin)
    
    return q_embed, k_embed

print(model.config)
print(model.config._attn_implementation)
print(model.model)
print(model.model.layers[0])

builder = ComputeGraphBuilder()
q = builder.input("q")
k = builder.input("k")
cos = builder.input("cos")
sin = builder.input("sin")
rope = llama_rope(builder, 4096, q, k, cos, sin)
