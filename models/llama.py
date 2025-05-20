from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from accelerate import init_empty_weights
import torch

from inference import ComputeGraphBuilder, ComputeGraph, ComputeGraphNode
from inference.name_scope import NameScope
from typing import Tuple

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

def just(b: ComputeGraphBuilder, x: int):
    return b.fixed(f"just_{x}", torch.Tensor([x]).int())

def rotary_embed(b: ComputeGraphBuilder, position_ids_node: ComputeGraphNode, inv_freq_node: ComputeGraphNode, attention_scaling_node: ComputeGraphNode) -> tuple[ComputeGraphNode, ComputeGraphNode]:
    """
    Implements the LlamaRotaryEmbedding forward pass using ComputeGraphBuilder.
    Mirrors LlamaRotaryEmbedding.forward(self, x, position_ids) but x is only used for dtype/device context in HF,
    and batch_size for inv_freq expansion is taken from position_ids.shape[0].

    Args:
        b: ComputeGraphBuilder instance.
        position_ids_node: Node representing position_ids [batch_size, seq_len].
        inv_freq_node: Node representing inv_freq [head_dim // 2].
        attention_scaling_node: Node representing attention_scaling (scalar).

    Returns:
        A tuple of (cos_node, sin_node).
    """
    # Define common dimension nodes for conciseness
    dim0_node = b.fixed("rotary_embed/dim0", torch.tensor([0], dtype=torch.long))
    dim1_node = b.fixed("rotary_embed/dim1", torch.tensor([1], dtype=torch.long))
    dim2_node = b.fixed("rotary_embed/dim2", torch.tensor([2], dtype=torch.long))
    cat_axis_node = b.fixed("rotary_embed/cat_axis", torch.tensor([-1], dtype=torch.long))

    pos_ids_shape_node = b.shape("rotary_embed/pos_ids_shape", position_ids_node)
    batch_size_scalar_node = b.index("rotary_embed/batch_size_scalar", pos_ids_shape_node, dim0_node)

    inv_freq_expanded = b.broadcast(
        "rotary_embed/inv_freq_expanded",
        b.unsqueeze("rotary_embed/inv_freq_u0_u2", 
                    b.unsqueeze("rotary_embed/inv_freq_u0", inv_freq_node, dim0_node), 
                    dim2_node), 
        dim0_node, 
        batch_size_scalar_node
    )

    position_ids_expanded = b.unsqueeze("rotary_embed/pos_ids_expanded", position_ids_node, dim1_node)

    freqs_matmul = b.matmul("rotary_embed/freqs_matmul", inv_freq_expanded, position_ids_expanded)
    freqs = b.transpose("rotary_embed/freqs_transposed", freqs_matmul, 1, 2)

    emb = b.cat("rotary_embed/emb_cat", freqs, freqs, cat_axis_node)

    cos_val = b.cos("rotary_embed/emb_cos", emb)
    cos_scaled = b.hadamard("rotary_embed/cos_scaled", cos_val, attention_scaling_node)

    sin_val = b.sin("rotary_embed/emb_sin", emb)
    sin_scaled = b.hadamard("rotary_embed/sin_scaled", sin_val, attention_scaling_node)
    return cos_scaled, sin_scaled

def apply_llama_rope(b: ComputeGraphBuilder, q: ComputeGraphNode, k: ComputeGraphNode, cos: ComputeGraphNode, sin: ComputeGraphNode):
    # Expected shapes:
    # q, k: [batch_size, heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim]

    # Create fixed nodes for constant indices and values
    fixed_zero = just(b,0)
    fixed_one = just(b,1)
    fixed_two = just(b,2)
    fixed_three = just(b, 3)
    fixed_neg_one_val = just(b,-1.0)

    # --- Derive dimensions dynamically ---
    q_shape = b.shape("q_shape", q)
    # num_heads = q.shape[0]
    num_heads_scalar = b.index("num_heads_scalar", q_shape, fixed_one)
    num_heads_node = num_heads_scalar
    # num_heads_node = b.unsqueeze("num_heads", num_heads_scalar, fixed_zero)  # Make it rank 1
    # seq_len = q.shape[1]
    seq_len_scalar = b.index("seq_len_scalar", q_shape, fixed_two)
    seq_len_node = seq_len_scalar
    # seq_len_node = b.unsqueeze("seq_len", seq_len_scalar, fixed_zero)  # Make it rank 1
    # head_dim = q.shape[2]
    head_dim_scalar = b.index("head_dim_scalar", q_shape, fixed_three)
    head_dim_node = head_dim_scalar
    # head_dim_node = b.unsqueeze("head_dim", head_dim_scalar, fixed_zero)  # Make it rank 1
    # head_dim_half = head_dim // 2
    head_dim_half_float = b.div("head_dim_half_float", head_dim_node, fixed_two)
    head_dim_half_node = b.floor("head_dim_half", head_dim_half_float)  # Convert to integer
    # --- End Derive dimensions ---

    def rotate_half(x: ComputeGraphNode, dim_idx_node: ComputeGraphNode, dim_node: ComputeGraphNode, dim_half_node: ComputeGraphNode):
        # Slice the tensor into two halves along the specified dimension index
        x1 = b.slice("x1", x, dim_idx_node, fixed_zero, dim_half_node)
        x2 = b.slice("x2", x, dim_idx_node, dim_half_node, dim_node)
        
        # Broadcast neg_one_val to match x2's shape by broadcasting along each dimension
        # we want to unsqueeze the neg_one enough times to match the required dim for the subsequent hadamard
        neg_one_unsqueeze = b.unsqueeze("neg_one_unsqueeze", fixed_neg_one_val, fixed_one)
        neg_one_unsqueeze = b.unsqueeze("neg_one_unsqueeze2", neg_one_unsqueeze, fixed_two)

        neg_one_broadcast = b.broadcast("neg_one_broadcast", neg_one_unsqueeze, fixed_zero, num_heads_node)
        neg_one_broadcast = b.broadcast("neg_one_broadcast2", neg_one_broadcast, fixed_one, seq_len_node)
        neg_one_broadcast = b.broadcast("neg_one_broadcast3", neg_one_broadcast, fixed_two, dim_half_node)
        
        # Negate the second half (x2)
        neg_x2 = b.hadamard("neg_x2", x2, neg_one_broadcast)
        
        # Concatenate the negated second half and the first half along the specified dimension index
        return b.cat("rotated", neg_x2, x1, dim_idx_node)
    
    # First unsqueeze cos and sin to [bsz, 1, seq_len, head_dim]
    cos_unsqueezed = b.unsqueeze("cos_unsqueezed", cos, fixed_one)
    sin_unsqueezed = b.unsqueeze("sin_unsqueezed", sin, fixed_one)
    
    # Broadcast to [bsz, heads, seq_len, head_dim]
    cos_broadcast = b.broadcast("cos_broadcast_heads", cos_unsqueezed, fixed_one, num_heads_node)
    sin_broadcast = b.broadcast("sin_broadcast_heads", sin_unsqueezed, fixed_one, num_heads_node)
    
    # Apply RoPE to query
    with NameScope.push_scope("rope_q"):
        q_rotated = rotate_half(q, fixed_three, head_dim_node, head_dim_half_node)
        q_cos = b.hadamard("cos", q, cos_broadcast)
        q_sin = b.hadamard("sin", q_rotated, sin_broadcast)
        q_embed = b.add("embed", q_cos, q_sin)
    
    # Apply RoPE to key
    with NameScope.push_scope("rope_k"):
        k_rotated = rotate_half(k, fixed_three, head_dim_node, head_dim_half_node)
        k_cos = b.hadamard("cos", k, cos_broadcast)
        k_sin = b.hadamard("sin", k_rotated, sin_broadcast)
        k_embed = b.add("embed", k_cos, k_sin)
    
    return q_embed, k_embed


def llama_attn(
    b: ComputeGraphBuilder, 
    hidden_states: ComputeGraphNode,
    head_dim: int,
    n_kv_heads: int,
    q_weight: ComputeGraphNode,
    k_weight: ComputeGraphNode,
    v_weight: ComputeGraphNode,
    o_weight: ComputeGraphNode,
    position_embeddings: Tuple[ComputeGraphNode, ComputeGraphNode], # (cosine, sine) 
    ):
    # we will assume that the hidden_states is of shape (seq_len, hidden_dim)
    input_shape  = b.shape("input_shape", hidden_states)
    with NameScope.push_scope("fixed values"):
        zero_node = just(b, 0)
        one_node  = just(b, 1)
        two_node  = just(b, 2)
    # assume the input shape is [batch_size, seq_len, hidden_dim]
    with NameScope.push_scope("input_shape_indices"):
        batch_size =  b.index("bsz", input_shape, zero_node) 
        seq_len    = b.index("seq_len", input_shape, one_node)
        hidden_dim = b.index("hidden_dim", input_shape, two_node)
    
    bsz_cat_seqlen = b.cat("bsz+seq_len", batch_size, seq_len, zero_node)
    head_dim_node = b.fixed("head_dim", torch.tensor([head_dim]))

    nhead_node_q  = b.div("nheads_q", hidden_dim, head_dim_node)
    nhead_node_kv = just(b, n_kv_heads)
    ngroups_node  = b.div("n_kv_groups", nhead_node_q, nhead_node_kv)
    
    # perform qkv projection
    with NameScope.push_scope("query_states"):
        hidden_shape = b.cat("bsz+seq_len+nheads", bsz_cat_seqlen, nhead_node_q, zero_node)
        hidden_shape = b.cat("bsz+seq_len+nheads+head_dim", hidden_shape, head_dim_node, zero_node)

        q_weight_unsqueezed = b.unsqueeze("q_weight_unsqueezed", q_weight, zero_node)
        q_proj = b.matmul("q_proj", hidden_states, q_weight_unsqueezed) # (bsz, seq_len, hidden_dim)
        q_proj = b.reshape("reshaped", q_proj, hidden_shape) # (bsz, seq_len, nheads_q, head_dim)
        query_states = b.transpose("transposed<1,2>", q_proj, 1, 2) # (bsz, nheads_q, seq_len, head_dim)

    hidden_shape = b.cat("bsz+seq_len+nheads", bsz_cat_seqlen, nhead_node_kv, zero_node)
    hidden_shape = b.cat("bsz+seq_len+nheads+head_dim", hidden_shape, head_dim_node, zero_node)
    
    with NameScope.push_scope("key_states"):
        k_weight_unsqueezed = b.unsqueeze("k_weight_unsqueezed", k_weight, zero_node) # (1, hidden_dim, head_dim)
        k_proj = b.matmul("k_proj", hidden_states, k_weight_unsqueezed) # (bsz, seq_len, hidden_dim)
        k_proj = b.reshape("reshaped", k_proj, hidden_shape) # (bsz, seq_len, nheads_kv, head_dim)
        key_states = b.transpose("transposed<1,2>", k_proj, 1, 2) # (bsz, nheads_kv, seq_len, head_dim)

        # hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        # return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        
        key_states = b.unsqueeze("unsqueezed", key_states, two_node) # (bsz, nheads_kv, 1, seq_len, head_dim)
        key_states = b.broadcast("bcasted", key_states, two_node, ngroups_node) # (bsz, nheads_kv, ngroups, seq_len, head_dim)
        
        # Calculate the proper output shape for reshape
        nheads_q_shape = b.hadamard("nheads_q_shape", nhead_node_kv, ngroups_node) # nheads_q = nheads_kv * ngroups

        batch_size_node = b.index("batch_size", input_shape, zero_node)
        reshape_shape = b.cat("reshape_shape_start", batch_size_node, nheads_q_shape, zero_node)
        reshape_shape = b.cat("reshape_shape_with_seq", reshape_shape, seq_len, zero_node)
        reshape_shape = b.cat("final_reshape_shape", reshape_shape, head_dim_node, zero_node)

        key_states = b.reshape("grouped", key_states, reshape_shape) # (bsz, nheads_q, seq_len, head_dim)
    
    with NameScope.push_scope("value_states"):
        v_weight_unsqueezed = b.unsqueeze("v_weight_unsqueezed", v_weight, zero_node)
        v_proj = b.matmul("v_proj", hidden_states, v_weight_unsqueezed)
        v_proj = b.reshape("reshaped", v_proj, hidden_shape)
        value_states = b.transpose("transposed<1,2>", v_proj, 1, 2)
        
        value_states = b.unsqueeze("unsqueezed", value_states, two_node) # (bsz, nheads_kv, 1, seq_len, head_dim)
        value_states = b.broadcast("bcasted", value_states, two_node, ngroups_node) # (bsz, nheads_kv, ngroups, seq_len, head_dim)
        
        # Calculate the proper output shape for reshape
        nheads_q_shape = b.hadamard("nheads_q_shape", nhead_node_kv, ngroups_node) # nheads_q = nheads_kv * ngroups
        
        batch_size_node = b.index("batch_size", input_shape, zero_node)
        reshape_shape = b.cat("reshape_shape_start", batch_size_node, nheads_q_shape, zero_node)
        reshape_shape = b.cat("reshape_shape_with_seq", reshape_shape, seq_len, zero_node)
        reshape_shape = b.cat("final_reshape_shape", reshape_shape, head_dim_node, zero_node)

        value_states = b.reshape("grouped", value_states, reshape_shape) # (bsz, nheads_q, seq_len, head_dim)
    
    with NameScope.push_scope("llama_rope"):
        query_states, key_states = apply_llama_rope(b, query_states, key_states, *position_embeddings)
    
    with NameScope.push_scope("attn_weights"):
        k_T = b.transpose("transposed<2,3>", key_states, 2, 3)
        attn_weights = b.matmul("matmul", query_states, k_T)
    
    attn_scores = b.softmax("softmax", attn_weights, dim=two_node) 
    with NameScope.push_scope("attn_output"):
        attn_out = b.matmul("matmul", attn_scores, value_states)
        attn_out = b.transpose("transposed<1,2>", attn_out, 1, 2)
        attn_out = b.reshape("reshaped", attn_out, input_shape)
    with NameScope.push_scope("output_states"):
        o_weight_unsqueezed = b.unsqueeze("o_weight_unsqueezed", o_weight, zero_node)
        attn_out = b.matmul("matmul", attn_out, o_weight_unsqueezed)
    return attn_out


def layernorm(
    b: ComputeGraphBuilder,
    hidden_states: ComputeGraphNode, # Should be of shape (batch_size, seq_len, hidden_dim)
    weight: ComputeGraphNode, # Should be of shape (hidden_size,)
    eps: ComputeGraphNode # Should be a scalar tensor
):
    fixed_zero = just(b, 0)
    fixed_one = just(b, 1)
    fixed_two = just(b, 2)
    fixed_neg_one = just(b, -1)

    # Get shape information for broadcasting
    hidden_states_shape = b.shape("hidden_states_shape_for_ln", hidden_states)
    batch_size_node = b.index("ln_batch_size", hidden_states_shape, fixed_zero)
    hidden_dim_node = b.index("ln_hid_dim", hidden_states_shape, fixed_two)
    seq_len_node = b.index("ln_seq_len", hidden_states_shape, fixed_one)

    hidden_states_pow_2 = b.square("hidden_states_pow_2", hidden_states)
    variance_no_keepdim = b.reduce_mean("variance_no_keepdim", hidden_states_pow_2, fixed_neg_one)
    variance = b.unsqueeze("variance_keepdim", variance_no_keepdim, fixed_neg_one) # Shape: [B, S, 1]

    # Prepare eps (scalar) for addition with variance ([B, S, 1])
    with NameScope.push_scope("eps_prep_for_add"):
        # eps is assumed to be rank 1 (shape [1]) from input in tests
        # eps_unsq0 = b.unsqueeze("unsq_to_rank1", eps, fixed_zero)      # Removed: eps starts as rank 1
        eps_unsq1 = b.unsqueeze("eps_unsq_to_rank2", eps, fixed_one)    # eps (shape [1]) -> eps_unsq1 (shape [1,1])
        eps_unsq_final = b.unsqueeze("eps_unsq_to_rank3", eps_unsq1, fixed_two) # eps_unsq1 (shape [1,1]) -> eps_unsq_final (shape [1,1,1])
        eps_bcast_bsz = b.broadcast("bcast_bsz", eps_unsq_final, fixed_zero, batch_size_node) # Shape: [B, 1, 1]
        eps_broadcasted = b.broadcast("bcast_seq", eps_bcast_bsz, fixed_one, seq_len_node)   # Shape: [B, S, 1]
    variance_plus_eps = b.add("variance_plus_eps", variance, eps_broadcasted)

    rsqrt_variance_plus_eps = b.rsqrt("rsqrt_variance_plus_eps", variance_plus_eps)
    bcasted_rsqrt = b.broadcast("bcasted_sqrt", rsqrt_variance_plus_eps, fixed_neg_one, hidden_dim_node)

    hidden_states_normalized = b.hadamard("hidden_states_normalized", hidden_states, bcasted_rsqrt)

    # Prepare weight ([H]) for Hadamard with hidden_states_normalized ([B, S, H])
    with NameScope.push_scope("weight_prep_for_scale"):
        weight_unsq_bsz = b.unsqueeze("unsq_dim0", weight, fixed_zero)             # Shape: [1, H]
        weight_bcast_bsz = b.broadcast("bcast_bsz", weight_unsq_bsz, fixed_zero, batch_size_node) # Shape: [B, H]
        weight_unsq_seq = b.unsqueeze("unsq_dim1", weight_bcast_bsz, fixed_one)      # Shape: [B, 1, H]
        weight_broadcasted = b.broadcast("bcast_seq", weight_unsq_seq, fixed_one, seq_len_node) # Shape: [B, S, H]
    scaled_hidden_states = b.hadamard("scaled_hidden_states", hidden_states_normalized, weight_broadcasted)
    
    return scaled_hidden_states

def mlp(b: ComputeGraphBuilder, x: ComputeGraphNode, weights: list[ComputeGraphNode]):
    # the idea here is that we stream the input through the weights
    out = x
    for idx, weight in enumerate(weights):
        out = b.matmul(f"layer{idx}", out, weight)
    return out

def llama_mlp(
    b: ComputeGraphBuilder,
    x: ComputeGraphNode,
    act: str,
    gate_proj: ComputeGraphNode,
    up_proj: ComputeGraphNode,
    down_proj: ComputeGraphNode,
):
    gate_result = b.matmul("gate_proj", x, gate_proj)
    up_result = b.matmul("up_proj" ,x, up_proj)
    if act == "silu":
        act_result = b.silu("silu", gate_result)
    
    mul = b.hadamard("act x up", act_result, up_result)
    down_result = b.matmul("down_proj", mul, down_proj)
    return down_result

def llama_fwd(
    b: ComputeGraphBuilder, hidden_states: ComputeGraphNode, 
    head_dim: int, n_kv_heads: int, mlp_act: str,
    weight_dict: dict[str, ComputeGraphNode], position_embeddings: Tuple[ComputeGraphNode, ComputeGraphNode]
):
    with NameScope.push_scope("layernorm"):
        layernormed = layernorm(b, hidden_states, **weight_dict["input_layernorm"])
    with NameScope.push_scope('attention'):
        attention  = llama_attn(b, layernormed, head_dim, n_kv_heads, position_embeddings=position_embeddings, **weight_dict["self_attn"])
    
    res_added = b.add("res_added", hidden_states, attention)
    post_layernorm = layernorm(b, res_added, **weight_dict["post_layernorm"])
    mlp_result = llama_mlp(b, post_layernorm, mlp_act, **weight_dict["mlp"])
    
    res_added_2 = b.add('res_added2', res_added, mlp_result) 
    return res_added_2


def llama_model(
    b: ComputeGraphBuilder, 
    tokens: ComputeGraphNode, # input tokens to the model; must have shape (bsz, seq_len)
    position_ids: ComputeGraphNode,
    weights: list[dict[str, ComputeGraphNode]] # dict per layer
):
    # weights[0] houses all statics
    # compute the embeddings of the input tokens

    dim0_node = b.fixed("embed_dim0", torch.tensor([0], dtype=torch.long))    
    embed_tokens = b.index_select("embed_tokens", weights[0]["embed_matrix"], dim0_node, tokens)
    cos_node, sin_node = rotary_embed(
        b, position_ids, weights[0]["inv_freq"], weights[0]["attn_scaling"]
    )

    layer_out = embed_tokens
    layer_out = b.debug("debug_hidden_0", layer_out)
    for layer_idx in range(1, len(weights)):
        with NameScope.push_scope(f"layer{layer_idx}"):
            layer_out = llama_fwd(
                b, layer_out, weights[0]["head_dim"], weights[0]["n_kv_heads"], weights[0]["mlp_act"],
                weights[layer_idx], (cos_node, sin_node)
            )
            layer_out = b.debug(f"debug_hidden_{layer_idx}", layer_out)
    
    layer_out = layernorm(b, layer_out, weights[0]["final_norm_weight"], weights[0]["final_norm_eps"])
    layer_out = b.debug("final_out", layer_out)
    return layer_out