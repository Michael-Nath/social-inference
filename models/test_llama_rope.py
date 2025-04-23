import torch
import numpy as np
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from inference.graph import (
    ComputeGraphBuilder, ComputeGraph, DEFAULT_NODE_OUTPUT,
    InputNode, OutputNode
)
from inference.simulator import simulate
from inference.pipeline import PartitionWork, InputAssignment, ComputePipeline, PipelineInput
from inference.tensor import Tensor
from inference.test_util import random_correlated_2d_tensor, llama_1b_cache
from .llama import apply_llama_rope

def test_llama_rope():
    # Test parameters
    batch_size = 1
    num_heads = 4
    seq_len = 8
    head_dim = 64  # Must be even for RoPE
    
    # Create random input tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    cos = torch.randn(batch_size, seq_len, head_dim)
    sin = torch.randn(batch_size, seq_len, head_dim)
    
    # Run Hugging Face implementation
    hf_q, hf_k = apply_rotary_pos_emb(q, k, cos, sin)
    
    # Run our implementation
    builder = ComputeGraphBuilder()
    
    # Create input nodes
    q_node = builder.input("q")
    k_node = builder.input("k")
    cos_node = builder.input("cos")
    sin_node = builder.input("sin")
    
    # Apply RoPE
    with builder.partition("p0"):
        q_embed, k_embed = apply_llama_rope(builder, head_dim, q_node, k_node, cos_node, sin_node)
    
    # Create output nodes
    builder.output("q_out", q_embed)
    builder.output("k_out", k_embed)
    
    # Build graph
    g = builder.build()
    print(g.validate_graph())

    # Create pipeline and enqueue inputs
    pipeline = ComputePipeline(g)
    inputs = {
        "q": Tensor.from_torch(q),
        "k": Tensor.from_torch(k),
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
    
    # Submit work result
    pipeline.submit_partition_work(result)
    
    # Get outputs
    output = pipeline.dequeue_output(blocking=False)
    assert output is not None
    
    q_out = output.outputs["q_out"].to_torch()
    k_out = output.outputs["k_out"].to_torch()
    
    # Compare results
    assert torch.allclose(q_out, hf_q, rtol=1e-3, atol=1e-3)
    assert torch.allclose(k_out, hf_k, rtol=1e-3, atol=1e-3) 