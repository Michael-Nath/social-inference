from .graph import ComputeGraph
from .simulator import simulate
from .pipeline import PartitionWork, InputAssignment
from .test_util import random_2d_tensor, llama_1b_cache

import numpy as np

def test_simulate_simple_matmul():
    g = ComputeGraph()
    x = g.input("x")
    y = g.input("y")
    with g.partition("p0"):
        z = g.matmul("z", x, y)
    g.output("o", z)
    g.freeze()

    ge = g.extract_partition("p0", include_cut_edges=False).encode()

    lhs = random_2d_tensor("1", (5, 5)).tensor
    rhs = random_2d_tensor("1", (5, 5)).tensor
    work = PartitionWork(
        correlation_id="1",
        partition="p0",
        graph=ge,
        inputs=[InputAssignment(node=z.name, input="lhs", tensor=lhs), InputAssignment(node=z.name, input="rhs", tensor=rhs)]
    )

    cache = llama_1b_cache()

    result = simulate(work, cache)
    assert result.correlation_id == work.correlation_id
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.node == z.name
    assert output.output == "output"
    assert np.allclose(output.tensor.to_numpy(), lhs.to_numpy() @ rhs.to_numpy())
