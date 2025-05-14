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

def test_index_reshape():
    g = ComputeGraphBuilder()
    # Input tensor: 2x3x4
    main_input = g.input("main_input_ir") 

    with g.partition("p0_index_reshape"):
        # IndexNode: Select the slice at index 1 along dim 0. Output should be 3x4.
        # For simplicity, our IndexNode will assume index is scalar and applies to dim 0.
        index_val = g.fixed("index_val_ir", torch.tensor([1], dtype=torch.int32))
        indexed_slice = g.index("indexed_op", main_input, index_val)

        # ReshapeNode: Reshape the 3x4 slice to 2x6
        reshape_dims1 = g.fixed("reshape_dims1_ir", torch.tensor([2, 6], dtype=torch.int32))
        reshaped_node1 = g.reshape("reshape_op1", indexed_slice, reshape_dims1)

        # ReshapeNode: Reshape the original 2x3x4 tensor to 6x4 (using -1)
        reshape_dims2 = g.fixed("reshape_dims2_ir", torch.tensor([-1, 4], dtype=torch.int32))
        reshaped_node2 = g.reshape("reshape_op2", main_input, reshape_dims2)
        
        # IndexNode: Select element at index [0,0] from reshaped_node1 (2x6 -> scalar)
        # This would require more complex indexing (e.g. index is [0,0] or multiple index nodes)
        # Let's simplify: index element at index 3 from a flattened version first.
        # Flatten reshaped_node1 (2x6 -> 12 elements)
        flatten_dims = g.fixed("flatten_dims_ir", torch.tensor([12], dtype=torch.int32))
        flattened_for_index = g.reshape("flatten_for_index_op", reshaped_node1, flatten_dims)
        scalar_index_val = g.fixed("scalar_index_val_ir", torch.tensor([3], dtype=torch.int32)) # 4th element
        indexed_element = g.index("indexed_element_op", flattened_for_index, scalar_index_val)

    g.output("indexed_slice_out", indexed_slice)
    g.output("reshaped1_out", reshaped_node1)
    g.output("reshaped2_out", reshaped_node2)
    g.output("indexed_element_out", indexed_element)
    
    graph = g.build()
    pipeline = ComputePipeline(graph)

    input_tensor_data = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    # main_input_ir: [[[ 0,  1,  2,  3], [ 4,  5,  6,  7], [ 8,  9, 10, 11]],
    #                 [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]

    # indexed_slice (main_input_ir[1]): [[12,13,14,15], [16,17,18,19], [20,21,22,23]] (shape 3x4)
    # reshaped_node1 (indexed_slice.reshape(2,6)): 
    #   [[12,13,14,15,16,17], [18,19,20,21,22,23]] (shape 2x6)
    # reshaped_node2 (main_input_ir.reshape(-1,4)): 
    #   [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19],[20,21,22,23]] (shape 6x4)
    # flattened_for_index (reshaped_node1.reshape(12)): 
    #   [12,13,14,15,16,17,18,19,20,21,22,23]
    # indexed_element (flattened_for_index[3]): 15 (scalar)

    pipeline.enqueue_input(PipelineInput(
        correlation_id="index_reshape_test_0",
        inputs={
            "main_input_ir": Tensor.from_torch(input_tensor_data),
        }
    ))
    return pipeline, graph

def test_transpose():
    g = ComputeGraphBuilder()
    input_tp = g.input("input_tp") # Transpose Input

    dim0_val = 1
    dim1_val = 2

    with g.partition("p0_transpose"):
        # TransposeNode constructor in worker.js will take dim0, dim1 from api_response
        # The ComputeGraphBuilder.transpose() should pass these.
        # No need for fixed tensors for dim0, dim1 if they are direct node attributes.
        transposed_node = g.transpose("transpose_op", input_tp, dim0=dim0_val, dim1=dim1_val)
    
    g.output("transpose_out", transposed_node)
    graph = g.build()
    pipeline = ComputePipeline(graph)

    # Input tensor: 2x3x4 (arange(24))
    # Transposing dim 1 and 2: expected output shape 2x4x3
    input_data = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    # PyTorch transpose for verification:
    # expected_output_torch = input_data.transpose(dim0_val, dim1_val)
    # print("Input for transpose:\n", input_data)
    # print(f"Expected output after transpose({dim0_val}, {dim1_val}):\n", expected_output_torch)

    pipeline.enqueue_input(PipelineInput(
        correlation_id="transpose_test_0",
        inputs={
            "input_tp": Tensor.from_torch(input_data),
        }
    ))
    return pipeline, graph

def test_trig_ops():
    g = ComputeGraphBuilder()
    trig_input = g.input("trig_input_val")

    with g.partition("p0_trig"):
        sin_node = g.sin("sin_op", trig_input)
        cos_node = g.cos("cos_op", trig_input)
        # Example: sin^2(x) + cos^2(x) = 1
        sin_squared = g.hadamard("sin_sq", sin_node, sin_node) # Using hadamard for square
        cos_squared = g.hadamard("cos_sq", cos_node, cos_node) # Using hadamard for square
        sum_sq = g.add("sum_sq_trig", sin_squared, cos_squared)

    g.output("sin_out", sin_node)
    g.output("cos_out", cos_node)
    g.output("sum_sq_out", sum_sq) # Should be close to 1
    graph = g.build()
    pipeline = ComputePipeline(graph)

    # Input values including 0, pi/2, pi, 3pi/2, 2pi
    # And some other values to check general computation
    angles = torch.tensor([[0, torch.pi/2, torch.pi, 3*torch.pi/2, 2*torch.pi],
                           [torch.pi/4, torch.pi/6, 1.0, -1.0, 2.5]], dtype=torch.float32)

    pipeline.enqueue_input(PipelineInput(
        correlation_id="trig_test_0",
        inputs={
            "trig_input_val": Tensor.from_torch(angles),
        }
    ))
    return pipeline, graph

def test_squared():
    g = ComputeGraphBuilder()
    sq_input = g.input("sq_input_val")

    with g.partition("p0_squared"):
        # Test squaring a mix of positive, negative, and zero values
        squared_node_float = g.square("squared_float", sq_input) # float32 input

        # Test with integer input
        int_input_fixed = g.fixed("int_fixed_sq", torch.tensor([-2, 0, 5, 100], dtype=torch.int32))
        squared_node_int = g.square("squared_int", int_input_fixed)
    
    g.output("squared_float_out", squared_node_float)
    g.output("squared_int_out", squared_node_int)
    graph = g.build()
    pipeline = ComputePipeline(graph)

    float_data = torch.tensor([-3.0, -0.5, 0.0, 1.5, 10.0], dtype=torch.float32)
    # Expected float: [9.0, 0.25, 0.0, 2.25, 100.0]
    # Expected int:   [4, 0, 25, 10000]

    pipeline.enqueue_input(PipelineInput(
        correlation_id="squared_test_0",
        inputs={
            "sq_input_val": Tensor.from_torch(float_data),
        }
    ))
    return pipeline, graph

def test_silu_rsqrt():
    g = ComputeGraphBuilder()
    silu_rsqrt_input = g.input("silu_rsqrt_in")

    with g.partition("p0_silu_rsqrt"):
        silu_node = g.silu("silu_op", silu_rsqrt_input)
        # Rsqrt requires positive inputs, so apply to abs(input) or an intermediate positive tensor.
        # For simplicity here, let's assume silu_rsqrt_input will be positive for rsqrt, or use abs.
        # PyTorch torch.rsqrt(torch.tensor([-4.0, 0.0, 4.0])) -> [nan, inf, 0.5]
        # We'll use a known positive tensor for rsqrt to avoid NaNs/Infs in simple test output.
        positive_fixed = g.fixed("positive_vals_rs", torch.tensor([1.0, 4.0, 9.0, 16.0, 0.01, 100.0], dtype=torch.float32))
        rsqrt_node = g.rsqrt("rsqrt_op", positive_fixed)
        rsqrt_from_silu_abs = g.rsqrt("rsqrt_silu_abs", g.hadamard("abs_silu", silu_node, silu_node)) # rsqrt(abs(silu(x))) is not rsqrt(x*sig(x))
                                                                                                # It's rsqrt( (x*sig(x))^2 ). A bit contrived.
                                                                                                # Let's make it rsqrt from absolute input: 
        abs_input_for_rsqrt = g.rsqrt("rsqrt_abs_input", g.hadamard("abs_in_val", silu_rsqrt_input, silu_rsqrt_input)) # rsqrt(x^2) = 1/|x|

    g.output("silu_out", silu_node)
    g.output("rsqrt_fixed_out", rsqrt_node) 
    g.output("rsqrt_abs_input_out", abs_input_for_rsqrt)
    graph = g.build()
    pipeline = ComputePipeline(graph)

    # Input values for SiLU: includes negative, zero, positive
    input_data = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=torch.float32)
    # SiLU(x) = x * sigmoid(x)
    # Rsqrt(positive_fixed) = [1.0, 0.5, 0.333..., 0.25, 10.0, 0.1]
    # Rsqrt(input_data^2) = 1/|input_data| = [0.2, 1.0, inf, 1.0, 0.2]

    pipeline.enqueue_input(PipelineInput(
        correlation_id="silu_rsqrt_test_0",
        inputs={
            "silu_rsqrt_in": Tensor.from_torch(input_data),
        }
    ))
    return pipeline, graph

def test_reduce_mean():
    g = ComputeGraphBuilder()
    rm_input = g.input("rm_input_val") # Reduce Mean input

    rm_input_3d = g.input("rm_input_3d_val")

    with g.partition("p0_reduce_mean"):
        # Test case 1: Reduce 2D tensor (3x4) along dim 1 (cols). Output should be (3,)
        dim_reduce1 = g.fixed("dim_reduce1_rm", torch.tensor([1], dtype=torch.int32))
        # Test case 2: Reduce same 2D tensor (3x4) along dim 0 (rows). Output should be (4,)
        dim_reduce0 = g.fixed("dim_reduce0_rm", torch.tensor([0], dtype=torch.int32))
        # Test case 3: Reduce 3D tensor (2x3x4) along dim 2. Output should be (2,3)
        dim_reduce2_3d = g.fixed("dim_reduce2_3d_rm", torch.tensor([2], dtype=torch.int32))

        reduced_node1 = g.reduce_mean("reduce_mean_op1", rm_input, dim_reduce1)
        reduced_node0 = g.reduce_mean("reduce_mean_op0", rm_input, dim_reduce0)
        reduced_node_3d = g.reduce_mean("reduce_mean_op3d", rm_input_3d, dim_reduce2_3d)

    g.output("reduce_mean1_out", reduced_node1)
    g.output("reduce_mean0_out", reduced_node0)
    g.output("reduce_mean3d_out", reduced_node_3d)
    graph = g.build()
    pipeline = ComputePipeline(graph)

    input_data_2d = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    # [[ 0,  1,  2,  3],
    #  [ 4,  5,  6,  7],
    #  [ 8,  9, 10, 11]]
    # reduce dim 1: [1.5, 5.5, 9.5]
    # reduce dim 0: [4., 5., 6., 7.]

    input_data_3d = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    # reduce dim 2:
    # [[[ 0,  1,  2,  3], -> 1.5
    #   [ 4,  5,  6,  7], -> 5.5
    #   [ 8,  9, 10, 11]], -> 9.5
    #  [[12, 13, 14, 15], -> 13.5
    #   [16, 17, 18, 19], -> 17.5
    #   [20, 21, 22, 23]]] -> 21.5
    # Result shape (2,3). Values: [[1.5, 5.5, 9.5], [13.5, 17.5, 21.5]]

    pipeline.enqueue_input(PipelineInput(
        correlation_id="reduce_mean_test_0",
        inputs={
            "rm_input_val": Tensor.from_torch(input_data_2d),
            "rm_input_3d_val": Tensor.from_torch(input_data_3d),
        }
    ))
    return pipeline, graph