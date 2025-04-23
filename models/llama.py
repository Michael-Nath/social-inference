from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from accelerate import init_empty_weights
import torch

from inference import ComputeGraphBuilder, ComputeGraph, ComputeGraphNode
from inference.name_scope import NameScope
from typing import Tuple

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


def llama_attn(
    b: ComputeGraphBuilder, 
    hidden_states: ComputeGraphNode, 
    head_dim: int,
    q_weight: ComputeGraphNode,
    k_weight: ComputeGraphNode,
    v_weight: ComputeGraphNode,
    o_weight: ComputeGraphNode,
    position_embeddings: Tuple[ComputeGraphNode, ComputeGraphNode], # (cosine, sine)
    ):
    input_shape  = b.shape("input_shape", hidden_states)
    # assume the input shape is [batch_size, seq_len, hidden_dim]
    with NameScope.push_scope("input_shape_indices"):
        batch_size =  b.index("bsz", input_shape, 0)
        seq_len    = b.index("seq_len", input_shape, 1)
        hidden_dim = b.index("hidden_dim", input_shape, 2)
    
    bsz_cat_seqlen = b.cat("bsz+seq_len", batch_size, seq_len, 0)
    bsz_cat_seqlen_cat_hidden_dim = b.cat("bsz+seq_len+hidden_dim", bsz_cat_seqlen, hidden_dim, 0)
    b.fixed([""])
    new_shape = b.cat("bsz+seq_len+hidden_dim")
    new_shape = b.cat("bsz+seq_len+hidden_dim", bsz_cat_seqlen, hidden_dim, 0)
    
    
    # perform qkv projection
    with NameScope.push_scope("query_states"):
        q_proj = b.matmul("q_proj", hidden_states, q_weight)
        q_proj = b.reshape("reshaped", q_proj, new_shape)
        query_states = b.transpose("transposed<1,2>", q_proj, 1, 2)

    with NameScope.push_scope("key_states"):
        k_proj = b.matmul("k_proj", hidden_states, k_weight)
        k_proj = b.reshape("reshaped", k_proj, new_shape)
        key_states = b.transpose("transposed<1,2>", k_proj, 1, 2)
    
    with NameScope.push_scope("value_states"):
        v_proj = b.matmul("v_proj", hidden_states, v_weight)
        v_proj = b.reshape("reshaped", v_proj, new_shape)
        value_states = b.transpose("transposed<1,2>", v_proj, 1, 2)
    
    query_states, key_states = apply_llama_rope(b, 1, query_states, key_states, *position_embeddings)
    
    with NameScope.push_scope("attn_weights"):
        k_T = b.transpose("transposed<2,3>", key_states, 2, 3)
        attn_weights = b.matmul("matmul", query_states, k_T)
    
    attn_scores = b.softmax(attn_weights, attn_weights, dim=2) 
    with NameScope.push_scope("attn_output"):
        attn_out = b.matmul("matmul", attn_scores, value_states)
        attn_out = b.transpose("transposed<1,2>", attn_out, 1, 2)
        attn_out = b.reshape("reshaped", attn_out, input_shape)
    with NameScope.push_scope("output_states"):
        attn_out = b.matmul("matmul", attn_out, o_weight)
    return attn_out
