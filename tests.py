import torch

from inference import (
    ComputePipeline, 
    PipelineInput, 
    Tensor, 
    ComputeGraphBuilder,
)

def simple_two_node():
    g = ComputeGraphBuilder()
    x = g.input("x")
    y = g.input("y")
    with g.partition("p0"):
        matmul = g.matmul("matmul", x, y)
        add = g.add("add1", matmul, x)
    z = g.output("output", add)
    return g.build()

def test_softmax():
    g = ComputeGraphBuilder()
    x = g.input("x")
    with g.partition("p0"):
        two = g.fixed("two", torch.tensor([2], dtype=torch.int32))
        softmax = g.softmax("softmax", x, two)
    y = g.output("output", softmax)
    g = g.build()

    pipeline = ComputePipeline(g)
    for i in range(20):
        pipeline.enqueue_input(PipelineInput(
            correlation_id=f"{i}",
            inputs={
                "x": Tensor.from_torch(torch.randn(4, 4, 4, 4)),
            },
        ))
    return pipeline, g

def test_unsqueeze():
    g = ComputeGraphBuilder()
    x = g.input("x")
    with g.partition("p0"):
        zero = g.fixed("zero", torch.tensor([0], dtype=torch.int32))
        unsqueeze = g.unsqueeze("unsqueeze", x, zero)
    y = g.output("output", unsqueeze)
    g = g.build()

    pipeline = ComputePipeline(g)
    for i in range(20):
        pipeline.enqueue_input(PipelineInput(
            correlation_id=f"{i}",
            inputs={
                "x": Tensor.from_torch(torch.randn(4, 4, 4, 4)),
            },
        ))
    return pipeline, g

def test_broadcast():
    g = ComputeGraphBuilder()
    x = g.input("x")
    with g.partition("p0"):
        two = g.fixed("two", torch.tensor([2], dtype=torch.int32))
        ten = g.fixed("ten", torch.tensor([10], dtype=torch.int32))
        unsqueezed = g.unsqueeze("unsqueeze", x, two)
        broadcast = g.broadcast("broadcast", unsqueezed, two, ten)
    y = g.output("output", broadcast)
    g = g.build()

    pipeline = ComputePipeline(g)
    for i in range(20):
        pipeline.enqueue_input(PipelineInput(
            correlation_id=f"{i}",
            inputs={
                "x": Tensor.from_torch(torch.randn(4, 4, 4, 4)),
            },
        ))
    return pipeline, g

def test_cat():
    g = ComputeGraphBuilder()
    a_in = g.input("a")
    b_in = g.input("b")
    with g.partition("p0"):
        # Concatenate along dimension 1
        dim_tensor = g.fixed("dim", torch.tensor([1], dtype=torch.int32))
        cat_node = g.cat("cat_op", a_in, b_in, dim_tensor)
    output = g.output("output", cat_node)
    graph = g.build()

    pipeline = ComputePipeline(graph)
    # Example input: two tensors of shape (2, 3, 4)
    # Expected output shape after cat along dim 1: (2, 6, 4)
    for i in range(5): # Let's create a few work items
        pipeline.enqueue_input(PipelineInput(
            correlation_id=f"cat_test_{i}",
            inputs={
                "a": Tensor.from_torch(torch.randn(2, 3, 4)),
                "b": Tensor.from_torch(torch.randn(2, 3, 4)),
            },
        ))
    return pipeline, graph

def test_math_ops():
    g = ComputeGraphBuilder()
    # Integer Test Path
    int_a_in = g.input("int_a")
    int_b_in = g.input("int_b")
    # Float Test Path
    float_a_in = g.input("float_a")
    float_b_in = g.input("float_b")

    with g.partition("p0_math_ops"):
        # Integer operations
        int_div_node = g.div("int_div", int_a_in, int_b_in)
        int_floor_node = g.floor("int_floor", int_div_node) # Floor of integer division
        int_ceil_node = g.ceil("int_ceil", int_div_node)   # Ceil of integer division

        # Float operations
        float_div_node = g.div("float_div", float_a_in, float_b_in)
        float_floor_node = g.floor("float_floor", float_div_node)
        float_ceil_node = g.ceil("float_ceil", float_div_node)

        # Mixed: floor a float, ceil a float, then div them
        # For this, let's use float_a_in directly for floor/ceil
        intermediate_float_floor = g.floor("intermediate_floor", float_a_in)
        intermediate_float_ceil = g.ceil("intermediate_ceil", float_a_in)
        mixed_div_node = g.div("mixed_div", intermediate_float_ceil, intermediate_float_floor) # e.g. ceil(7.5)/floor(7.5) = 8/7

    # Outputs
    g.output("int_floor_out", int_floor_node)
    g.output("int_ceil_out", int_ceil_node)
    g.output("float_floor_out", float_floor_node)
    g.output("float_ceil_out", float_ceil_node)
    g.output("mixed_div_out", mixed_div_node)
    
    graph = g.build()
    pipeline = ComputePipeline(graph)

    # Enqueue test inputs
    # Integers: 7 / 3 = 2.33... floor=2, ceil=3 (Note: div outputs float)
    # Floats: 7.5 / 2.5 = 3.0. floor=3, ceil=3
    # Mixed div: float_a = 7.5. ceil(7.5)=8, floor(7.5)=7. 8/7 = 1.14...
    for i in range(10):
        pipeline.enqueue_input(PipelineInput(
            correlation_id=f"math_ops_test_{i}",
            inputs={
                "int_a": Tensor.from_torch(torch.tensor([[7, 10]], dtype=torch.int32)),
                "int_b": Tensor.from_torch(torch.tensor([[3, 4]], dtype=torch.int32)),
                "float_a": Tensor.from_torch(torch.tensor([[7.5, 10.8]], dtype=torch.float32)),
                "float_b": Tensor.from_torch(torch.tensor([[2.5, 2.0]], dtype=torch.float32)),
            },
        ))
    return pipeline, graph

def test_hadamard():
    g = ComputeGraphBuilder()
    a_in = g.input("a_hadamard")
    b_in = g.input("b_hadamard")

    with g.partition("p0_hadamard"):
        hadamard_node = g.hadamard("hadamard_op", a_in, b_in)
    
    g.output("hadamard_out", hadamard_node)
    graph = g.build()
    pipeline = ComputePipeline(graph)

    # Example inputs
    tensor_a_data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    tensor_b_data = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=torch.float32)
    # Expected output: [[2.0, 6.0, 12.0], [20.0, 30.0, 42.0]]

    for i in range(5):
        pipeline.enqueue_input(PipelineInput(
            correlation_id=f"hadamard_test_{i}",
            inputs={
                "a_hadamard": Tensor.from_torch(tensor_a_data + i), # Add i for slight variation
                "b_hadamard": Tensor.from_torch(tensor_b_data + i),
            },
        ))
    return pipeline, graph

def test_gpu_division():
    g = ComputeGraphBuilder()
    a_in = g.input("a_div_gpu")
    b_in = g.input("b_div_gpu")

    with g.partition("p0_gpu_div"):
        # This node should be heavy enough to be scheduled on GPU
        # if DivNode supports GPU and threshold (default 1000) is met.
        # Input elements = 64 * 64 = 4096.
        div_node = g.div("heavy_div_op", a_in, b_in)
    
    g.output("div_out_gpu", div_node)
    graph = g.build()
    pipeline = ComputePipeline(graph)

    tensor_a_data = torch.randn(64, 64, dtype=torch.float32)
    # Ensure b_in does not contain zeros for simplicity
    tensor_b_data = torch.rand(64, 64, dtype=torch.float32) + 0.1 

    # Optional: if ComputePipeline supports checking expected outputs
    # expected_output_torch = tensor_a_data / tensor_b_data

    pipeline.enqueue_input(PipelineInput(
        correlation_id="gpu_div_test_0",
        inputs={
            "a_div_gpu": Tensor.from_torch(tensor_a_data),
            "b_div_gpu": Tensor.from_torch(tensor_b_data),
        },
        # expected_outputs={ 
        #     "div_out_gpu": Tensor.from_torch(expected_output_torch)
        # }
    ))
    return pipeline, graph