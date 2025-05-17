from transformers import LlamaConfig
from inference import ComputeGraphBuilder
from inference.name_scope import NameScope
import torch
from .utils import load_model, prepare_llama_model_statics, package_llama_decoder_layer_weights
from inference.tensor import Tensor
from inference.pipeline import ComputePipeline, PipelineInput
from inference.simulator import simulate
from inference.test_util import llama_1b_cache
from .llama import rotary_embed, llama_model

torch.set_grad_enabled(False)
MODEL_PATH = "meta-llama/Llama-3.2-1B"

def test_rotary_embed():
    print("\nTesting rotary_embed_implementation...")
    config = LlamaConfig.from_pretrained(MODEL_PATH)
    # Get inv_freq and attention_scaling from Transformers' rotary embedding utilities
    try:
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        # Create a dummy RoPE module to get parameters correctly initialized by HF
        hf_rope_module = LlamaRotaryEmbedding(config, device=None)
        inv_freq_torch = hf_rope_module.inv_freq.clone()
        attention_scaling_torch = torch.tensor(hf_rope_module.attention_scaling)
    except ImportError:
        raise ImportError("Could not import LlamaRotaryEmbedding to get test parameters.")

    b = ComputeGraphBuilder()
    # Define dynamic input node for position_ids before the partition block
    # as InputNodes don't belong to a specific compute partition.
    position_ids_node = b.input("test/position_ids")

    with b.partition("p0") :
        batch_size = 1
        seq_len = 5
        
        # Fixed nodes for inv_freq and attention_scaling remain in the partition
        inv_freq_node = b.fixed("test/inv_freq", inv_freq_torch)
        attention_scaling_node = b.fixed("test/attention_scaling", attention_scaling_torch)

        # Pass the input node to rotary_embed
        cos_node, sin_node = rotary_embed(b, position_ids_node, inv_freq_node, attention_scaling_node)

    b.output("test/cos_output", cos_node)
    b.output("test/sin_output", sin_node)
    graph = b.build()

    # Prepare the actual tensor for position_ids
    position_ids_torch_runtime = torch.arange(0, seq_len).unsqueeze(0).float()

    # HF comparison (using the runtime tensor for position_ids)
    dummy_x_for_hf = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.float32)
    cos_hf, sin_hf = hf_rope_module(dummy_x_for_hf, position_ids_torch_runtime)

    # Simulation
    pipeline = ComputePipeline(graph)
    # Provide position_ids as a dynamic input
    pipeline_inputs = {
        "test/position_ids": Tensor.from_torch(position_ids_torch_runtime)
    }
    pipeline.enqueue_input(PipelineInput(correlation_id="test_rotary_impl", inputs=pipeline_inputs))
    work = pipeline.get_partition_work("p0")
    assert work is not None, "No work generated for partition p0"
    
    result = simulate(work, {}) # Empty cache as inv_freq and attention_scaling are FixedNodes
    
    our_cos, our_sin = None, None
    for out_assignment in result.outputs:
        # Output node names are now prefixed with the partition name by the graph builder convention if they originate from a partition
        if out_assignment.node == cos_node.name: 
          our_cos = out_assignment.tensor.to_torch()
        if out_assignment.node == sin_node.name:
          our_sin = out_assignment.tensor.to_torch()

    assert our_cos is not None and our_sin is not None, "Outputs not found in simulation results. Check output node names."
    assert torch.allclose(cos_hf, our_cos, atol=1e-6), "Custom rotary cos output differs from HF"
    assert torch.allclose(sin_hf, our_sin, atol=1e-6), "Custom rotary sin output differs from HF"
    print("test_rotary_embed_implementation: PASSED")


def test_llama_model():
  model = load_model(MODEL_PATH)
  model.model = model.model.float()
  model.model = model.model.cpu()

  input_tokens = torch.randint(0, 1000, (1,)).cpu().unsqueeze(0)
  position_ids = torch.arange(0, 1).cpu().unsqueeze(0).float()
   
  b = ComputeGraphBuilder()
  input_tokens_node = b.input("input_tokens")
  pos_ids_node = b.input("position_ids")
  with b.partition("p0"):
    with NameScope.push_scope("statics"):
      statics = prepare_llama_model_statics(model, b)
      nodes = [statics]
  
      for layer_idx in range(16):
        with NameScope.push_scope(f"layer{layer_idx}"):
          prefix = f"model.layers.{layer_idx}."
          layer_weights = package_llama_decoder_layer_weights(model.model.layers[layer_idx], b, prefix, MODEL_PATH)
          nodes.append(layer_weights)

    our_out = llama_model(b, input_tokens_node, pos_ids_node, nodes)
  
  b.output("llama_out", our_out)

  # Build graph
  g = b.build()

  # Create pipeline and enqueue inputs
  pipeline = ComputePipeline(g)
  inputs = {
    "input_tokens": Tensor.from_torch(input_tokens),
    "position_ids": Tensor.from_torch(position_ids)
  }

  pipeline.enqueue_input(PipelineInput(correlation_id="test", inputs=inputs))
  work = pipeline.get_partition_work("p0")
  assert work is not None

  # Run simulation
  cache = llama_1b_cache()
  result = simulate(work, cache)

  our_output = None
  for output in result.outputs:
      if output.node == our_out.name:
          our_output = output.tensor.to_torch()
          break
  gt_output = model.model(input_tokens).last_hidden_state
  # Check if outputs match with tolerance
  assert torch.allclose(our_output, gt_output, atol=1e-4)
  print("Llama Model Pass passed successfully :)") 