from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
import torch
from inference import ComputeGraphBuilder

def load_model(path: str):
  model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.float16,
    device_map="auto"
  )
  return model

def prepare_llama_model_statics(model, b: ComputeGraphBuilder) -> dict:
    """
    Fetches static configuration values and prepares static graph nodes for a Llama model.
    The caller is responsible for setting the partition context on the ComputeGraphBuilder.

    Args:
        model: The loaded Hugging Face Llama model.
        b: ComputeGraphBuilder instance to create FixedNodes.

    Returns:
        A dictionary containing scalar static values (head_dim, n_kv_heads, mlp_act)
        and FixedNodes (inv_freq, attn_scaling, embed_matrix).
    """
    config = model.config

    # Fetch scalar static values
    all_statics = {
      "head_dim": config.head_dim,
      "n_kv_heads": config.num_key_value_heads,
      "mlp_act": config.hidden_act
    }

    # Prepare tensor-based static nodes using the builder passed by the caller
    # The caller must ensure 'b' is in the desired partition context.
    try:
        hf_rope_module = LlamaRotaryEmbedding(config, device=None)
        inv_freq_torch = hf_rope_module.inv_freq.clone()
        attention_scaling_torch = torch.tensor(hf_rope_module.attention_scaling)
    except ImportError:
        raise ImportError("Could not import LlamaRotaryEmbedding for RoPE parameters.")

    all_statics["inv_freq"] = b.fixed("static/inv_freq", inv_freq_torch)
    all_statics["attn_scaling"] = b.fixed("static/attn_scaling", attention_scaling_torch)
    
    embed_matrix_data = model.model.embed_tokens.weight.data.clone()
    all_statics["embed_matrix"] = b.fixed("static/embed_matrix", embed_matrix_data)

    # Fetch final layernorm weights and epsilon (from model.model.norm)
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        final_norm_weight_data = model.model.norm.weight.data.clone()
        all_statics["final_norm_weight"] = b.fixed("static/final_norm.weight", final_norm_weight_data)
        
        final_norm_eps_torch = torch.tensor(model.model.norm.variance_epsilon, dtype=torch.float32)
        all_statics["final_norm_eps"] = b.fixed("static/final_norm.eps", final_norm_eps_torch.unsqueeze(0))
    else:
        # This case should ideally not be hit if 'model' is a standard LlamaForCausalLM
        print("Warning: model.model.norm not found. Final layernorm parameters not fetched.")
    
    return all_statics

def package_llama_decoder_layer_weights(layer: LlamaDecoderLayer, b: ComputeGraphBuilder, prefix: str) -> dict:
    """
    Extracts weights from a PyTorch LlamaDecoderLayer and packages them as graph nodes.
    """
    packaged_weights = {
        "input_layernorm": {},
        "self_attn": {},
        "mlp": {},
        "post_layernorm": {}
    }
    all_param_nodes = {} # Temporary dict to hold nodes by their original HF names

    # Create graph nodes for all learnable parameters
    for name, param in layer.named_parameters():
        graph_node_name = f"weights/{name.replace('.', '/')}"
        data = param.data.clone()
        if len(data.shape) == 2:
            data = data.T
        complete_name = prefix + name
        node = b.fixed(complete_name, data.detach())
        all_param_nodes[complete_name] = node

    # Handle input layernorm
    if "input_layernorm.weight" in all_param_nodes:
        packaged_weights["input_layernorm"]["weight"] = all_param_nodes["input_layernorm.weight"]
    ln_eps_tensor = torch.tensor(layer.input_layernorm.variance_epsilon, dtype=torch.float32)
    packaged_weights["input_layernorm"]["eps"] = b.fixed("params/input_layernorm.eps", ln_eps_tensor.unsqueeze(0))


    # Handle self-attention weights
    # Mapping from graph function expected keys to Hugging Face parameter names
    attn_key_map = {
        "q_weight": "self_attn.q_proj.weight",
        "k_weight": "self_attn.k_proj.weight",
        "v_weight": "self_attn.v_proj.weight",
        "o_weight": "self_attn.o_proj.weight",
    }
    for graph_key, hf_key in attn_key_map.items():
        if hf_key in all_param_nodes:
            packaged_weights["self_attn"][graph_key] = all_param_nodes[hf_key]

    # Handle MLP weights
    mlp_key_map = {
        "gate_proj": "mlp.gate_proj.weight",
        "up_proj": "mlp.up_proj.weight",
        "down_proj": "mlp.down_proj.weight",
    }
    for graph_key, hf_key in mlp_key_map.items():
        if hf_key in all_param_nodes:
            packaged_weights["mlp"][graph_key] = all_param_nodes[hf_key]

    # Handle post-attention layernorm
    if "post_attention_layernorm.weight" in all_param_nodes:
        packaged_weights["post_layernorm"]["weight"] = all_param_nodes["post_attention_layernorm.weight"]
    post_ln_eps_tensor = torch.tensor(layer.post_attention_layernorm.variance_epsilon, dtype=torch.float32)
    packaged_weights["post_layernorm"]["eps"] = b.fixed("params/post_attention_layernorm.eps", post_ln_eps_tensor.unsqueeze(0))

    return packaged_weights