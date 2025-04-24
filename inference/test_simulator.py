import torch

from inference.tensor import Tensor
from .graph import (
    ComputeGraphBuilder, ComputeGraph, DEFAULT_NODE_OUTPUT, IndexNode,
    MatmulNode, SliceNode, UnsqueezeNode, BroadcastNode,
    CatNode, FixedNode, HadamardNode, ShapeNode, SoftmaxNode, DivNode,
    FloorNode, CeilNode
)
from .simulator import simulate
from .pipeline import PartitionWork, InputAssignment
from .test_util import random_correlated_2d_tensor, llama_1b_cache

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
    assert torch.allclose(output.tensor.to_torch(), lhs.to_torch() @ rhs.to_torch())

def test_simulate_slice():
    builder = ComputeGraphBuilder()
    x = builder.input("x")
    with builder.partition("p0"):
        # Slice a 2D tensor along dimension 0, taking elements 1 to 3
        # Parameters are now inputs via FixedNode
        dim_node = builder.fixed("dim_val", torch.tensor([0]))
        start_node = builder.fixed("start_val", torch.tensor([1]))
        end_node = builder.fixed("end_val", torch.tensor([3]))
        z = builder.slice("z", x, dim=dim_node, start=start_node, end=end_node)
    o = builder.output("o", z)
    g = builder.build()

    ge = g.extract_partition("p0", include_cut_edges=False).encode()

    input_tensor = random_correlated_2d_tensor("1", (5, 5)).tensor
    work = PartitionWork(
        correlation_id="1",
        partition="p0",
        graph=ge,
        # Only the main input tensor is needed now
        inputs=[InputAssignment(node=z.name, input=SliceNode.INPUT, tensor=input_tensor)]
    )

    cache = llama_1b_cache()
    result = simulate(work, cache)
    
    assert result.correlation_id == work.correlation_id
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.node == z.name
    assert output.output == DEFAULT_NODE_OUTPUT
    expected = input_tensor.to_torch()[1:3, :]
    assert torch.allclose(output.tensor.to_torch(), expected)

def test_simulate_unsqueeze():
    builder = ComputeGraphBuilder()
    x = builder.input("x")
    with builder.partition("p0"):
        # Unsqueeze a 2D tensor along dimension 1
        # dim is now an input via FixedNode
        dim_node = builder.fixed("dim_val", torch.tensor([1]))
        z = builder.unsqueeze("z", x, dim=dim_node)
    o = builder.output("o", z)
    g = builder.build()

    ge = g.extract_partition("p0", include_cut_edges=False).encode()

    input_tensor = random_correlated_2d_tensor("1", (5, 5)).tensor
    work = PartitionWork(
        correlation_id="1",
        partition="p0",
        graph=ge,
        # Only the main input tensor is needed now
        inputs=[InputAssignment(node=z.name, input=UnsqueezeNode.INPUT, tensor=input_tensor)]
    )

    cache = llama_1b_cache()
    result = simulate(work, cache)
    
    assert result.correlation_id == work.correlation_id
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.node == z.name
    assert output.output == DEFAULT_NODE_OUTPUT
    expected = input_tensor.to_torch().unsqueeze(1)
    assert torch.allclose(output.tensor.to_torch(), expected)

def test_simulate_broadcast():
    builder = ComputeGraphBuilder()
    x = builder.input("x")
    with builder.partition("p0"):
        # Broadcast a 2D tensor along dimension 1 to size 3
        # dim and n are now inputs via FixedNode
        dim_node = builder.fixed("dim_val", torch.tensor([1]))
        n_node = builder.fixed("n_val", torch.tensor([3]))
        z = builder.broadcast("z", x, dim=dim_node, n=n_node)
    o = builder.output("o", z)
    g = builder.build()

    ge = g.extract_partition("p0", include_cut_edges=False).encode()

    input_tensor = random_correlated_2d_tensor("1", (5, 1)).tensor
    work = PartitionWork(
        correlation_id="1",
        partition="p0",
        graph=ge,
        # Only the main input tensor is needed now
        inputs=[InputAssignment(node=z.name, input=BroadcastNode.INPUT, tensor=input_tensor)]
    )

    cache = llama_1b_cache()
    result = simulate(work, cache)
    
    assert result.correlation_id == work.correlation_id
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.node == z.name
    assert output.output == DEFAULT_NODE_OUTPUT
    expected = input_tensor.to_torch().expand(-1, 3)
    assert torch.allclose(output.tensor.to_torch(), expected)

def test_simulate_cat():
    builder = ComputeGraphBuilder()
    x = builder.input("x")
    y = builder.input("y")
    with builder.partition("p0"):
        # Concatenate two 2D tensors along dimension 1
        # dim is now an input via FixedNode
        dim_node = builder.fixed("dim_val", torch.tensor([1]))
        z = builder.cat("z", x, y, dim=dim_node)
    o = builder.output("o", z)
    g = builder.build()

    ge = g.extract_partition("p0", include_cut_edges=False).encode()

    input1 = random_correlated_2d_tensor("1", (5, 3)).tensor
    input2 = random_correlated_2d_tensor("2", (5, 2)).tensor
    work = PartitionWork(
        correlation_id="1",
        partition="p0",
        graph=ge,
        # Only the data inputs are needed now
        inputs=[
            InputAssignment(node=z.name, input=CatNode.A, tensor=input1),
            InputAssignment(node=z.name, input=CatNode.B, tensor=input2)
        ]
    )

    cache = llama_1b_cache()
    result = simulate(work, cache)
    
    assert result.correlation_id == work.correlation_id
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.node == z.name
    assert output.output == DEFAULT_NODE_OUTPUT
    expected = torch.cat([input1.to_torch(), input2.to_torch()], dim=1)
    assert torch.allclose(output.tensor.to_torch(), expected)

def test_simulate_fixed():
    builder = ComputeGraphBuilder()
    with builder.partition("p0"):
        # Create a fixed tensor node
        fixed_tensor = torch.ones(3, 3)
        z = builder.fixed("z", fixed_tensor)
    o = builder.output("o", z)
    g = builder.build()

    ge = g.extract_partition("p0", include_cut_edges=False).encode()

    work = PartitionWork(
        correlation_id="1",
        partition="p0",
        graph=ge,
        inputs=[]  # No inputs needed for fixed node
    )

    cache = llama_1b_cache()
    result = simulate(work, cache)
    
    assert result.correlation_id == work.correlation_id
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.node == z.name
    assert output.output == DEFAULT_NODE_OUTPUT
    assert torch.allclose(output.tensor.to_torch(), fixed_tensor)

def test_simulate_hadamard():
    builder = ComputeGraphBuilder()
    x = builder.input("x")
    y = builder.input("y")
    with builder.partition("p0"):
        # Element-wise multiplication of two tensors
        z = builder.hadamard("z", x, y)
    o = builder.output("o", z)
    g = builder.build()

    ge = g.extract_partition("p0", include_cut_edges=False).encode()

    input1 = random_correlated_2d_tensor("1", (5, 5)).tensor
    input2 = random_correlated_2d_tensor("2", (5, 5)).tensor
    work = PartitionWork(
        correlation_id="1",
        partition="p0",
        graph=ge,
        inputs=[
            InputAssignment(node=z.name, input=HadamardNode.A, tensor=input1),
            InputAssignment(node=z.name, input=HadamardNode.B, tensor=input2)
        ]
    )

    cache = llama_1b_cache()
    result = simulate(work, cache)
    
    assert result.correlation_id == work.correlation_id
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.node == z.name
    assert output.output == DEFAULT_NODE_OUTPUT
    expected = input1.to_torch() * input2.to_torch()
    assert torch.allclose(output.tensor.to_torch(), expected)

def test_simulate_softmax():
    builder = ComputeGraphBuilder()
    x = builder.input("x")
    with builder.partition("p0"):
        # Apply softmax along dimension 1
        # dim is now an input via FixedNode
        dim_node = builder.fixed("dim_val", torch.tensor([1]))
        z = builder.softmax("z", x, dim=dim_node)
    o = builder.output("o", z)
    g = builder.build()

    ge = g.extract_partition("p0", include_cut_edges=False).encode()

    input_tensor = random_correlated_2d_tensor("1", (5, 5)).tensor
    work = PartitionWork(
        correlation_id="1",
        partition="p0",
        graph=ge,
        # Only the main input tensor is needed now
        inputs=[InputAssignment(node=z.name, input=SoftmaxNode.INPUT, tensor=input_tensor)]
    )

    cache = llama_1b_cache()
    result = simulate(work, cache)
    
    assert result.correlation_id == work.correlation_id
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.node == z.name
    assert output.output == DEFAULT_NODE_OUTPUT
    expected = torch.nn.functional.softmax(input_tensor.to_torch(), dim=1)
    assert torch.allclose(output.tensor.to_torch(), expected)

def test_simulate_index():
    builder = ComputeGraphBuilder()
    x = builder.input("x")
    idx = builder.input("idx")
    with builder.partition("p0"):
        # Index into a tensor using another tensor as indices
        z = builder.index("z", x, idx)
    o = builder.output("o", z)
    g = builder.build()

    ge = g.extract_partition("p0", include_cut_edges=False).encode()

    # Create a 3D tensor for testing
    input_tensor = Tensor.from_torch(torch.randn(5,4,3))
    
    # Create an index tensor with valid indices
    index_tensor = Tensor.from_torch(torch.tensor([2, 3]))
    
    work = PartitionWork(
        correlation_id="1",
        partition="p0",
        graph=ge,
        inputs=[
            InputAssignment(node=z.name, input=IndexNode.INPUT, tensor=input_tensor),
            InputAssignment(node=z.name, input=IndexNode.INDEX, tensor=index_tensor)
        ]
    )

    cache = llama_1b_cache()
    result = simulate(work, cache)
    
    assert result.correlation_id == work.correlation_id
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.node == z.name
    assert output.output == DEFAULT_NODE_OUTPUT
    
    # Expected result is the value at indices [2, 3] in the input tensor
    expected = input_tensor.to_torch()[2, 3]
    assert torch.allclose(output.tensor.to_torch(), expected)

def test_simulate_shape():
    builder = ComputeGraphBuilder()
    x = builder.input("x")
    with builder.partition("p0"):
        # Get the shape of the input tensor
        z = builder.shape("z", x)
    o = builder.output("o", z)
    g = builder.build()

    ge = g.extract_partition("p0", include_cut_edges=False).encode()

    # Create a 3D tensor for testing
    input_tensor = Tensor.from_torch(torch.randn(5,4,3))
    
    work = PartitionWork(
        correlation_id="1",
        partition="p0",
        graph=ge,
        inputs=[InputAssignment(node=z.name, input=ShapeNode.INPUT, tensor=input_tensor)]
    )

    cache = llama_1b_cache()
    result = simulate(work, cache)
    
    assert result.correlation_id == work.correlation_id
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.node == z.name
    assert output.output == DEFAULT_NODE_OUTPUT
    
    # Expected result is the shape of the input tensor as a 1D tensor
    expected = torch.tensor(input_tensor.to_torch().shape, dtype=torch.long)
    assert torch.allclose(output.tensor.to_torch(), expected)

def test_simulate_div():
    builder = ComputeGraphBuilder()
    x = builder.input("x")
    y = builder.input("y")
    with builder.partition("p0"):
        # Element-wise division of two tensors
        z = builder.div("z", x, y)
    o = builder.output("o", z)
    g = builder.build()

    ge = g.extract_partition("p0", include_cut_edges=False).encode()

    input1 = random_correlated_2d_tensor("1", (5, 5)).tensor
    input2 = random_correlated_2d_tensor("2", (5, 5)).tensor
    work = PartitionWork(
        correlation_id="1",
        partition="p0",
        graph=ge,
        inputs=[
            InputAssignment(node=z.name, input=DivNode.A, tensor=input1),
            InputAssignment(node=z.name, input=DivNode.B, tensor=input2)
        ]
    )

    cache = llama_1b_cache()
    result = simulate(work, cache)
    
    assert result.correlation_id == work.correlation_id
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.node == z.name
    assert output.output == DEFAULT_NODE_OUTPUT
    expected = input1.to_torch() / input2.to_torch()
    assert torch.allclose(output.tensor.to_torch(), expected)

def test_simulate_shape_index_unsqueeze():
    builder = ComputeGraphBuilder()
    x = builder.input("x")
    with builder.partition("p0"):
        # Get shape of input tensor
        shape = builder.shape("shape", x)
        # Extract first dimension (should be a scalar)
        dim0 = builder.index("dim0", shape, builder.fixed("zero_idx", torch.tensor([0])))
        # Unsqueeze to make it rank 1
        dim0_unsqueezed = builder.unsqueeze("dim0_unsqueezed", dim0, builder.fixed("zero_dim", torch.tensor([0])))
    o = builder.output("o", dim0_unsqueezed)
    g = builder.build()

    ge = g.extract_partition("p0", include_cut_edges=False).encode()

    # Create a 3D tensor for testing
    input_tensor = Tensor.from_torch(torch.randn(5,4,3))
    
    work = PartitionWork(
        correlation_id="1",
        partition="p0",
        graph=ge,
        inputs=[InputAssignment(node=shape.name, input=ShapeNode.INPUT, tensor=input_tensor)]
    )

    cache = llama_1b_cache()
    result = simulate(work, cache)
    
    assert result.correlation_id == work.correlation_id
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.node == dim0_unsqueezed.name
    assert output.output == DEFAULT_NODE_OUTPUT
    
    # Expected result is the first dimension of the input tensor as a 1D tensor
    expected = torch.tensor([input_tensor.to_torch().shape[0]])
    assert torch.allclose(output.tensor.to_torch(), expected)

def test_simulate_floor_ceil():
    builder = ComputeGraphBuilder()
    x = builder.input("x")
    with builder.partition("p0"):
        # Test floor and ceil operations
        floor_result = builder.floor("floor", x)
        ceil_result = builder.ceil("ceil", x)
    builder.output("floor_out", floor_result)
    builder.output("ceil_out", ceil_result)
    g = builder.build()

    ge = g.extract_partition("p0", include_cut_edges=False).encode()

    # Create input tensor with floating point values
    input_tensor = Tensor.from_torch(torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float32))
    
    work = PartitionWork(
        correlation_id="1",
        partition="p0",
        graph=ge,
        inputs=[
            InputAssignment(node=floor_result.name, input=FloorNode.INPUT, tensor=input_tensor),
            InputAssignment(node=ceil_result.name, input=CeilNode.INPUT, tensor=input_tensor)
        ]
    )

    cache = llama_1b_cache()
    result = simulate(work, cache)
    
    assert result.correlation_id == work.correlation_id
    assert len(result.outputs) == 2
    
    # Check floor output
    floor_output = next(o for o in result.outputs if o.node == floor_result.name)
    assert floor_output.output == DEFAULT_NODE_OUTPUT
    expected_floor = torch.tensor([-2, -1, 0, 1], dtype=torch.long)
    assert torch.allclose(floor_output.tensor.to_torch(), expected_floor)
    
    # Check ceil output
    ceil_output = next(o for o in result.outputs if o.node == ceil_result.name)
    assert ceil_output.output == DEFAULT_NODE_OUTPUT
    expected_ceil = torch.tensor([-1, 0, 1, 2], dtype=torch.long)
    assert torch.allclose(ceil_output.tensor.to_torch(), expected_ceil)
