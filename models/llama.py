from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from accelerate import init_empty_weights
import torch

from inference import ComputeGraphBuilder, ComputeGraph, ComputeGraphNode
from inference.name_scope import NameScope

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

def apply_llama_rope(b: ComputeGraphBuilder, dim: int, q: ComputeGraphNode, k: ComputeGraphNode, cos: ComputeGraphNode, sin: ComputeGraphNode, unsqueeze_dim: int = 1):
    def rotate_half(x: ComputeGraphNode):
        x1 = b.slice("x1", x, 3, 0, dim // 2)
        x2 = b.slice("x2", x, 3, dim // 2, dim)
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
    with NameScope.push_scope("rope_q"):
        q_rotated = rotate_half(q)
        q_cos = b.hadamard("cos", q, cos_broadcast)
        q_sin = b.hadamard("sin", q_rotated, sin_broadcast)
        # Add the two terms together
        q_embed = b.add("embed", q_cos, q_sin)
    
    # Apply RoPE to key
    with NameScope.push_scope("rope_k"):
        k_rotated = rotate_half(k)
        k_cos = b.hadamard("cos", k, cos_broadcast)
        k_sin = b.hadamard("sin", k_rotated, sin_broadcast)
        # Add the two terms together
        k_embed = b.add("embed", k_cos, k_sin)
    
    return q_embed, k_embed