from .graph import ComputeGraphBuilder, ComputeGraph, DEFAULT_NODE_OUTPUT, MatmulNode
from .simulator import simulate
from .pipeline import PartitionWork, InputAssignment
from .test_util import random_correlated_2d_tensor, llama_1b_cache

import numpy as np

def test_simulate_simple_matmul():
    builder = ComputeGraphBuilder()
    x = builder.input("x")
    y = builder.input("y")
    with builder.partition("p0"):
        z = builder.matmul("z", x, y)
    o = builder.output("o", z)
    g = builder.build()

    ge = g.extract_partition("p0", include_cut_edges=False).encode()

    lhs = random_correlated_2d_tensor("1", (5, 5)).tensor
    rhs = random_correlated_2d_tensor("1", (5, 5)).tensor
    work = PartitionWork(
        correlation_id="1",
        partition="p0",
        graph=ge,
        inputs=[InputAssignment(node=z.name, input=MatmulNode.LHS, tensor=lhs), InputAssignment(node=z.name, input=MatmulNode.RHS, tensor=rhs)]
    )

    cache = llama_1b_cache()

    result = simulate(work, cache)
    assert result.correlation_id == work.correlation_id
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.node == z.name
    assert output.output == DEFAULT_NODE_OUTPUT
    assert np.allclose(output.tensor.to_numpy(), lhs.to_numpy() @ rhs.to_numpy())
