from models.llama import llama_causal
from models.utils import prepare_llama_model_statics, package_llama_decoder_layer_weights, layer_params
from inference import ComputeGraphBuilder
from transformers import AutoConfig
from inference import NameScope

MODEL_PATH = "meta-llama/Llama-3.2-1B"

def build_llaam_causal_mp():
  b = ComputeGraphBuilder() 
  # prepare the weights 
  config = AutoConfig.from_pretrained(MODEL_PATH)
  input_tokens_node = b.input("input_tokens")
  pos_ids_node = b.input("position_ids")
  
  # create a partition per layer

  outer_part  = "pre_and_post"
  with b.partition(outer_part):
    with NameScope.push_scope("statics"):
      statics = prepare_llama_model_statics(config, b)
      nodes = [statics]

  for p_idx in range(16):
    with b.partition(f"layer_{p_idx}"):
      layer_idx = p_idx
      with NameScope.push_scope(f"layer{p_idx}"):
        prefix = f"model.layers.{layer_idx}."
        layer_weights = package_llama_decoder_layer_weights(layer_params, b, prefix, MODEL_PATH)
        nodes.append(layer_weights)
    
  layer_parts = [f"layer_{i}" for i in range(16)]
  our_out = llama_causal(b, input_tokens_node, pos_ids_node, nodes, outer_part, layer_parts) 

  with b.partition(outer_part):
    b.output("llama_out", our_out)
  g = b.build()
  return g