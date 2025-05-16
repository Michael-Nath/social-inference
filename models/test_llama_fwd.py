import torch
import numpy as np
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm, LlamaDecoderLayer, LlamaRotaryEmbedding
from transformers.models.llama.configuration_llama import LlamaConfig
from .llama import llama_attn, layernorm, llama_fwd
from .utils import load_model, prepare_llama_model_statics, package_llama_decoder_layer_weights
from inference.tensor import Tensor
from inference.test_util import llama_1b_cache
from inference.simulator import simulate
from inference.graph import ComputeGraphBuilder
from inference.pipeline import ComputePipeline, PipelineInput


def test_llama_layernorm():
    # fetch the weights from a layernorm module of llama 
    batch_size = 1
    seq_len = 1
    hidden_dim = 2048
    rms_norm_eps = 1e-5
    layernorm_module = LlamaRMSNorm(hidden_size=hidden_dim, eps=rms_norm_eps)

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

    # hardcode the weights
    layer_weights = torch.randn(hidden_dim)
    layernorm_module.weight.data = layer_weights.clone()

    with torch.no_grad():
        gt_output = layernorm_module(hidden_states)
        
    builder = ComputeGraphBuilder()
    # Create input nodes
    hidden_states_node = builder.input("hidden_states")
    eps_node = builder.input("eps")
    weight_node = builder.input("weights")

    with builder.partition("p0"):
        layernorm_node = layernorm(builder, hidden_states_node, weight_node, eps_node)
    builder.output("layernorm_output", layernorm_node)

    # Build graph
    g = builder.build()

    # Create pipeline and enqueue inputs
    pipeline = ComputePipeline(g)
    inputs = {
        "hidden_states": Tensor.from_torch(hidden_states),
        "eps": Tensor.from_torch(torch.tensor([rms_norm_eps])),
        "weights": Tensor.from_torch(layer_weights)
    }
    pipeline.enqueue_input(PipelineInput(correlation_id="test", inputs=inputs))

    # Get partition work
    work = pipeline.get_partition_work("p0")
    assert work is not None
    
    # Run simulation
    cache = llama_1b_cache()
    result = simulate(work, cache)
    our_output = result.outputs[0].tensor.to_torch()
    assert torch.allclose(gt_output, our_output, rtol=1e-3)

        

def test_llama_attn():
    # Test parameters
    batch_size = 1
    num_heads = 1
    seq_len = 1
    head_dim = 64
    hidden_dim = num_heads * head_dim

    # Input
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    cos = torch.randn(1, seq_len, head_dim)
    sin = torch.randn(1, seq_len, head_dim)

    # Hardcoded weights
    mock_q_weight = torch.randn(hidden_dim, hidden_dim)
    mock_k_weight = torch.randn(hidden_dim, hidden_dim)
    mock_v_weight = torch.randn(hidden_dim, hidden_dim)
    mock_o_weight = torch.randn(hidden_dim, hidden_dim)

    # Create a real LlamaAttention object
    config = LlamaConfig(hidden_size=hidden_dim, num_attention_heads=num_heads)
    attn = LlamaAttention(config, 0)

    # Replace projection weights with hardcoded ones
    attn.q_proj.weight.data = mock_q_weight.clone()
    attn.k_proj.weight.data = mock_k_weight.clone()
    attn.v_proj.weight.data = mock_v_weight.clone()
    attn.o_proj.weight.data = mock_o_weight.clone()

    # Forward pass
    with torch.no_grad():
        gt_output, _ = attn(
            hidden_states=hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None
        )

    # Run our implementation
    builder = ComputeGraphBuilder()
    
    # Create input nodes
    hidden_states_node = builder.input("hidden_states")
    q_weight = builder.input("q_weight")
    k_weight = builder.input("k_weight")
    v_weight = builder.input("v_weight")
    o_weight = builder.input("o_weight")
    cos_node = builder.input("cos")
    sin_node = builder.input("sin")
    
    # Apply RoPE
    with builder.partition("p0"):
        attn_output = llama_attn(builder,
            hidden_states_node, head_dim, q_weight=q_weight, k_weight=k_weight, v_weight=v_weight, o_weight=o_weight, n_kv_heads=num_heads, position_embeddings=(cos_node, sin_node)
        )
    
    # Create output nodes
    builder.output("attn_output", attn_output)
    
    # Build graph
    g = builder.build()

    # Create pipeline and enqueue inputs
    pipeline = ComputePipeline(g)
    inputs = {
        "hidden_states": Tensor.from_torch(hidden_states),
        "q_weight": Tensor.from_torch(mock_q_weight.T),
        "k_weight": Tensor.from_torch(mock_k_weight.T),
        "v_weight": Tensor.from_torch(mock_v_weight.T),
        "o_weight": Tensor.from_torch(mock_o_weight.T),
        "cos": Tensor.from_torch(cos),
        "sin": Tensor.from_torch(sin)
    }
    pipeline.enqueue_input(PipelineInput(correlation_id="test", inputs=inputs))

    # Get partition work
    work = pipeline.get_partition_work("p0")
    assert work is not None
    
    # Run simulation
    cache = llama_1b_cache()
    result = simulate(work, cache)
    our_attn_output = result.outputs[0].tensor.to_torch()
    
    # Check if tensors are close
    is_close = torch.allclose(gt_output, our_attn_output, rtol=1e-2)
    
    if not is_close:
        # Calculate absolute difference
        abs_diff = torch.abs(gt_output - our_attn_output)
        # Find max difference and its location
        max_diff = torch.max(abs_diff)
        max_diff_idx = torch.argmax(abs_diff.flatten())
        max_diff_pos = np.unravel_index(max_diff_idx.item(), abs_diff.shape)
        
        # Calculate relative difference
        rel_diff = abs_diff / (torch.abs(gt_output) + 1e-8)
        max_rel_diff = torch.max(rel_diff)
        max_rel_diff_idx = torch.argmax(rel_diff.flatten())
        max_rel_diff_pos = np.unravel_index(max_rel_diff_idx.item(), rel_diff.shape)
        
        print(f"Tensors not close! Max absolute diff: {max_diff.item():.6f} at position {max_diff_pos}")
        print(f"Max relative diff: {max_rel_diff.item():.6f} at position {max_rel_diff_pos}")
        print(f"Values at max abs diff: GT={gt_output[max_diff_pos].item():.6f}, Ours={our_attn_output[max_diff_pos].item():.6f}")
        print(f"Values at max rel diff: GT={gt_output[max_rel_diff_pos].item():.6f}, Ours={our_attn_output[max_rel_diff_pos].item():.6f}")
        
        # Show a small region around the max difference
        r, c, d = max_diff_pos
        r_start, r_end = max(0, r-1), min(abs_diff.shape[0], r+2)
        c_start, c_end = max(0, c-1), min(abs_diff.shape[1], c+2)
        d_start, d_end = max(0, d-1), min(abs_diff.shape[2], d+2)
        
        print("\nRegion around max absolute difference:")
        print("Ground truth:")
        print(gt_output[r_start:r_end, c_start:c_end, d_start:d_end])
        print("Our output:")
        print(our_attn_output[r_start:r_end, c_start:c_end, d_start:d_end])
        
    assert is_close, "Attention outputs don't match within tolerance"


def layer_correct(model_name, model, idx: int):
    batch_size = 1
    seq_len = 1
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads


    head_dim = model.config.head_dim
    hidden_dim = num_heads * head_dim
    # create some hidden states
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    cos = torch.randn(1, seq_len, head_dim)
    sin = torch.randn(1, seq_len, head_dim)

    # Extract the first decoder layer
    first_decoder_layer: LlamaDecoderLayer = model.model.layers[idx].cpu()
    b = ComputeGraphBuilder()
    with b.partition("p0"):
        # Use the new utility function to package weights
        prefix = f"model.layers.{idx}."
        packaged_layer_weights = package_llama_decoder_layer_weights(first_decoder_layer, b, prefix, model_name)
        
        # Prepare the specific weight_dict for the current llama_fwd function
        weight_dict_for_llama_fwd = {
            "input_layernorm": packaged_layer_weights["input_layernorm"],
            "self_attn": packaged_layer_weights["self_attn"],
            "mlp": packaged_layer_weights["mlp"],
            "post_layernorm": packaged_layer_weights["post_layernorm"]
        }

        # Create a graph node for the hidden_states input tensor
        hidden_states_node = b.input("hidden_states")

        # Now you can call your graph implementation of llama_fwd
        cos_node = b.fixed("cos", cos)
        sin_node = b.fixed("sin", sin)
        fwd_output_node = llama_fwd(
            b, hidden_states_node, head_dim, num_kv_heads, model.config.hidden_act, weight_dict_for_llama_fwd,
            (cos_node, sin_node) 
        )
    b.output("llama_fwd_out", fwd_output_node)

    # Build graph
    g = b.build()

    # Create pipeline and enqueue inputs
    pipeline = ComputePipeline(g)
    inputs = {
        "hidden_states": Tensor.from_torch(hidden_states),
    }
    pipeline.enqueue_input(PipelineInput(correlation_id="test", inputs=inputs))
    work = pipeline.get_partition_work("p0")
    assert work is not None

    # Run simulation
    cache = llama_1b_cache()
    result = simulate(work, cache)

    our_output = None
    for output in result.outputs:
        if output.node == fwd_output_node.name:
            our_output = output.tensor.to_torch()
            break
    gt_output = first_decoder_layer(hidden_states, position_embeddings = (cos, sin))[0]

    # Check if outputs match with tolerance
    assert torch.allclose(our_output, gt_output, atol=1e-4)
    print(f"Layer {idx} passed successfully!")
    return True
    

def test_llama_fwd():
    
    model_path = "meta-llama/Llama-3.2-1B"
    # Load the model
    model = load_model(model_path)
    model.model = model.model.float()

    # Prepare static parameters and nodes using the utility function
    builder_for_statics = ComputeGraphBuilder()
    with builder_for_statics.partition("statics"):
        all_static_params = prepare_llama_model_statics(model, builder_for_statics)

    # Separate scalar and node statics for clarity if needed, though all_static_params holds both
    scalar_static_params = {
        k: v for k, v in all_static_params.items() 
        if not hasattr(v, 'name') # Assuming nodes have a .name attribute, scalars don't
    }
    global_static_nodes = {
        k: v for k, v in all_static_params.items() 
        if hasattr(v, 'name')
    }

    print(f"Fetched scalar statics: {scalar_static_params}")
    print(f"Packaged global static nodes: { {k: v.name for k, v in global_static_nodes.items()} }")

    for idx in range(len(model.model.layers)):
        # layer_correct currently tests llama_fwd per layer and handles its own weight packaging.
        # If testing the full llama_model, scalar_static_params and global_static_nodes would be used here.
        assert layer_correct(model_path, model, idx )

if __name__ == "__main__":
    test_llama_attn()