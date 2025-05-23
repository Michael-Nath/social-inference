from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
import torch
from inference import ComputeGraphBuilder

def load_model(path: str):
  model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.float32,
    device_map="auto"
  )
  return model


# Layer 0
layer_params = [
    'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
    'self_attn.o_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight',
    'mlp.down_proj.weight', 'input_layernorm.weight', 'post_attention_layernorm.weight'
]

model_params = [layer_params]
        
def prepare_llama_model_statics(config, b: ComputeGraphBuilder) -> dict:
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

    # Fetch scalar static values
    all_statics = {
      "head_dim": config.head_dim,
      "n_kv_heads": config.num_key_value_heads,
      "mlp_act": config.hidden_act
    }

    hf_rope_module = LlamaRotaryEmbedding(config, device=None)
    inv_freq_torch = hf_rope_module.inv_freq.clone()
    attention_scaling_torch = torch.tensor(hf_rope_module.attention_scaling).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1,1,64)

    all_statics["inv_freq"] = b.fixed("inv_freq", inv_freq_torch)
    all_statics["attn_scaling"] = b.fixed("attn_scaling", attention_scaling_torch)
    
    embed_matrix = b.safetensor("embed_matrix", "meta-llama/Llama-3.2-1B", "model.embed_tokens.weight")
    all_statics["embed_matrix"] = embed_matrix

    # Fetch final layernorm weights and epsilon (from model.model.norm)
    final_norm_weight = b.safetensor("final_norm.weight", "meta-llama/Llama-3.2-1B", "model.norm.weight")
    all_statics["final_norm_weight"] = final_norm_weight
    final_norm_eps_torch = torch.tensor(1e-5, dtype=torch.float32)
    all_statics["final_norm_eps"] = b.fixed("final_norm.eps", final_norm_eps_torch.unsqueeze(0))
    
    
    return all_statics

def package_llama_decoder_layer_weights(layer: list[str], b: ComputeGraphBuilder, prefix: str, model_name: str) -> dict:
    packaged_weights = {
        "input_layernorm": {},
        "self_attn": {},
        "mlp": {},
        "post_layernorm": {}
    }
    all_param_nodes = {} # Temporary dict to hold nodes by their original HF names

    # Create graph nodes for all learnable parameters
    for name in layer:
        tensor_name = prefix + name
        complete_name = model_name + tensor_name
        node = b.safetensor(complete_name, model_name, tensor_name)
        all_param_nodes[tensor_name] = node

    # Handle input layernorm
    if f"{prefix}input_layernorm.weight" in all_param_nodes:
        packaged_weights["input_layernorm"]["weight"] = all_param_nodes[f"{prefix}input_layernorm.weight"]
    ln_eps_tensor = torch.tensor(1e-5, dtype=torch.float32)
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
        mod_hf_key = f"{prefix}{hf_key}"
        if mod_hf_key in all_param_nodes:
            packaged_weights["self_attn"][graph_key] = all_param_nodes[mod_hf_key]

    # Handle MLP weights
    mlp_key_map = {
        "gate_proj": "mlp.gate_proj.weight",
        "up_proj": "mlp.up_proj.weight",
        "down_proj": "mlp.down_proj.weight",
    }
    for graph_key, hf_key in mlp_key_map.items():
        mod_hf_key = f"{prefix}{hf_key}"
        if mod_hf_key in all_param_nodes:
            packaged_weights["mlp"][graph_key] = all_param_nodes[mod_hf_key]

    # Handle post-attention layernorm
    if f"{prefix}post_attention_layernorm.weight" in all_param_nodes:
        packaged_weights["post_layernorm"]["weight"] = all_param_nodes[f"{prefix}post_attention_layernorm.weight"]
    post_ln_eps_tensor = torch.tensor(1e-5, dtype=torch.float32)
    packaged_weights["post_layernorm"]["eps"] = b.fixed("params/post_attention_layernorm.eps", post_ln_eps_tensor.unsqueeze(0))
    return packaged_weights