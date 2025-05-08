import torch
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.llama.configuration_llama import LlamaConfig
from .llama import llama_attn
from inference.tensor import Tensor
from inference.test_util import llama_1b_cache
from inference.simulator import simulate
from inference.graph import (
    ComputeGraphBuilder, ComputeGraph, DEFAULT_NODE_OUTPUT,
    InputNode, OutputNode
)
from inference.pipeline import PartitionWork, InputAssignment, ComputePipeline, PipelineInput

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

    # Remove bias (if needed, to match test expectations)
    # attn.q_proj.bias.data.zero_()
    # attn.k_proj.bias.data.zero_()
    # attn.v_proj.bias.data.zero_()
    # attn.o_proj.bias.data.zero_()

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
            hidden_states_node, head_dim, q_weight, k_weight, v_weight, o_weight, (cos_node, sin_node)
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
    assert torch.allclose(gt_output, our_attn_output, rtol=1e-3)

if __name__ == "__main__":
    test_llama_attn()
