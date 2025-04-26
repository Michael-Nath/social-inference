from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from accelerate import init_empty_weights
import torch

from inference import ComputeGraphBuilder, ComputeGraph, ComputeGraphNode
from inference.name_scope import NameScope
from typing import Tuple

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

def just(b: ComputeGraphBuilder, x: int):
    return b.fixed(f"just_{x}", torch.Tensor([x]))

def apply_llama_rope(b: ComputeGraphBuilder, q: ComputeGraphNode, k: ComputeGraphNode, cos: ComputeGraphNode, sin: ComputeGraphNode):
    # Expected shapes:
    # q, k: [heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim]

    # Create fixed nodes for constant indices and values
    fixed_zero = just(b,0)
    fixed_one = just(b,1)
    fixed_two = just(b,2)
    fixed_neg_one_val = just(b,-1.0)

    # --- Derive dimensions dynamically ---
    q_shape = b.shape("q_shape", q)
    # num_heads = q.shape[0]
    num_heads_scalar = b.index("num_heads_scalar", q_shape, fixed_zero)
    num_heads_node = b.unsqueeze("num_heads", num_heads_scalar, fixed_zero)  # Make it rank 1
    # seq_len = q.shape[1]
    seq_len_scalar = b.index("seq_len_scalar", q_shape, fixed_one)
    seq_len_node = b.unsqueeze("seq_len", seq_len_scalar, fixed_zero)  # Make it rank 1
    # head_dim = q.shape[2]
    head_dim_scalar = b.index("head_dim_scalar", q_shape, fixed_two)
    head_dim_node = b.unsqueeze("head_dim", head_dim_scalar, fixed_zero)  # Make it rank 1
    # head_dim_half = head_dim // 2
    head_dim_half_float = b.div("head_dim_half_float", head_dim_node, fixed_two_val)
    head_dim_half_node = b.floor("head_dim_half", head_dim_half_float)  # Convert to integer
    # --- End Derive dimensions ---

    def rotate_half(x: ComputeGraphNode, dim_idx_node: ComputeGraphNode, dim_node: ComputeGraphNode, dim_half_node: ComputeGraphNode):
        # Slice the tensor into two halves along the specified dimension index
        x1 = b.slice("x1", x, dim_idx_node, fixed_zero, dim_half_node)
        x2 = b.slice("x2", x, dim_idx_node, dim_half_node, dim_node)
        
        # Broadcast neg_one_val to match x2's shape by broadcasting along each dimension
        neg_one_broadcast = b.broadcast("neg_one_broadcast", fixed_neg_one_val, fixed_zero, num_heads_node)
        neg_one_broadcast = b.broadcast("neg_one_broadcast2", neg_one_broadcast, fixed_one, seq_len_node)
        neg_one_broadcast = b.broadcast("neg_one_broadcast3", neg_one_broadcast, fixed_two, dim_half_node)
        
        # Negate the second half (x2)
        neg_x2 = b.hadamard("neg_x2", x2, neg_one_broadcast)
        
        # Concatenate the negated second half and the first half along the specified dimension index
        return b.cat("rotated", neg_x2, x1, dim_idx_node)
    
    # First unsqueeze cos and sin to [1, seq_len, head_dim]
    cos_unsqueezed = b.unsqueeze("cos_unsqueezed", cos, fixed_zero)
    sin_unsqueezed = b.unsqueeze("sin_unsqueezed", sin, fixed_zero)
    
    # Broadcast to [heads, seq_len, head_dim]
    cos_broadcast = b.broadcast("cos_broadcast_heads", cos_unsqueezed, fixed_zero, num_heads_node)
    sin_broadcast = b.broadcast("sin_broadcast_heads", sin_unsqueezed, fixed_zero, num_heads_node)
    
    # Apply RoPE to query
    with NameScope.push_scope("rope_q"):
        q_rotated = rotate_half(q, fixed_two, head_dim_node, head_dim_half_node)
        q_cos = b.hadamard("cos", q, cos_broadcast)
        q_sin = b.hadamard("sin", q_rotated, sin_broadcast)
        q_embed = b.add("embed", q_cos, q_sin)
    
    # Apply RoPE to key
    with NameScope.push_scope("rope_k"):
        k_rotated = rotate_half(k, fixed_two, head_dim_node, head_dim_half_node)
        k_cos = b.hadamard("cos", k, cos_broadcast)
        k_sin = b.hadamard("sin", k_rotated, sin_broadcast)
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
        batch_size =  b.index("bsz", input_shape, b.fixed("0", torch.tensor([0])))
        seq_len    = b.index("seq_len", input_shape, b.fixed("1", torch.tensor([1])))
        hidden_dim = b.index("hidden_dim", input_shape, b.fixed("2", torch.tensor([2])))
    
    bsz_cat_seqlen = b.cat("bsz+seq_len", batch_size, seq_len, 0)
    head_dim_node = b.fixed("head_dim", torch.tensor([head_dim]))
    nhead_node = b.div("nheads", hidden_dim, head_dim_node)

    hidden_shape = b.cat("bsz+seq_len+nheads", bsz_cat_seqlen, nhead_node, 0)
    hidden_shape = b.cat("bsz+seq_len+nheads+head_dim", hidden_shape, head_dim_node, 0)
    
    # perform qkv projection
    with NameScope.push_scope("query_states"):
        q_proj = b.matmul("q_proj", hidden_states, q_weight)
        q_proj = b.reshape("reshaped", q_proj, hidden_shape)
        query_states = b.transpose("transposed<1,2>", q_proj, 1, 2)

    with NameScope.push_scope("key_states"):
        k_proj = b.matmul("k_proj", hidden_states, k_weight)
        k_proj = b.reshape("reshaped", k_proj, hidden_shape)
        key_states = b.transpose("transposed<1,2>", k_proj, 1, 2)
    
    with NameScope.push_scope("value_states"):
        v_proj = b.matmul("v_proj", hidden_states, v_weight)
        v_proj = b.reshape("reshaped", v_proj, hidden_shape)
        value_states = b.transpose("transposed<1,2>", v_proj, 1, 2)
    
    query_states, key_states = apply_llama_rope(b, 1, query_states, key_states, *position_embeddings)
    
    with NameScope.push_scope("attn_weights"):
        k_T = b.transpose("transposed<2,3>", key_states, 2, 3)
        attn_weights = b.matmul("matmul", query_states, k_T)
    
    attn_scores = b.softmax("softmax", attn_weights, dim=2) 
    with NameScope.push_scope("attn_output"):
        attn_out = b.matmul("matmul", attn_scores, value_states)
        attn_out = b.transpose("transposed<1,2>", attn_out, 1, 2)
        attn_out = b.reshape("reshaped", attn_out, input_shape)
    with NameScope.push_scope("output_states"):
        attn_out = b.matmul("matmul", attn_out, o_weight)
    return attn_out