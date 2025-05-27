from models.llama import llama_causal
from models.utils import prepare_llama_model_statics, package_llama_decoder_layer_weights, layer_params
from inference import ComputeGraphBuilder
from transformers import AutoConfig
from inference import NameScope
import torch

def build_llaam_causal_mp(MODEL_PATH):
  b = ComputeGraphBuilder() 
  # prepare the weights 
  config = AutoConfig.from_pretrained(MODEL_PATH)
  input_tokens_node = b.input("input_tokens")
  pos_ids_node = b.input("position_ids")
  
  # create a partition per layer

  with b.partition("pre"):
    with NameScope.push_scope("statics_pre"):
      statics = prepare_llama_model_statics(config, MODEL_PATH, b)
  
  with b.partition("post"):
    with NameScope.push_scope("statics_post"):
      statics["final_norm_weight_post"] = b.safetensor("final_norm.weight", MODEL_PATH, "model.norm.weight")
      final_norm_eps_torch = torch.tensor(config.rms_norm_eps, dtype=torch.float32) 
      statics["final_norm_eps_post"] = b.fixed("final_norm.eps", final_norm_eps_torch.unsqueeze(0))
      statics["embed_matrix_post"] = b.safetensor("embed_matrix", MODEL_PATH, "model.embed_tokens.weight")

  nodes = [statics]
  num_layers = config.num_hidden_layers
  #num_layers = 1

  for p_idx in range(num_layers):
    with b.partition(f"layer{p_idx}"):
      layer_idx = p_idx
      with NameScope.push_scope(f"layer{p_idx}"):
        prefix = f"model.layers.{layer_idx}."
        layer_weights = package_llama_decoder_layer_weights(layer_params, b, prefix, MODEL_PATH)
        nodes.append(layer_weights)
    
  layer_parts = [f"layer{i}" for i in range(num_layers)]
  our_out = llama_causal(b, input_tokens_node, pos_ids_node, nodes, layer_parts) 
  b.output("llama_out", our_out)
  
  # with b.partition("post"):
  #   b.output("llama_out", our_out)
  g = b.build()
  return g