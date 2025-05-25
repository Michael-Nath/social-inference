from .graph import ComputeGraphBuilder, ComputeGraph, MatmulNode
from .pipeline import ComputePipeline, PipelineInput, PipelineOutput
from .simulator import simulate
from .test_util import random_2d_tensor, llama_1b_cache

def test_simple_pipeline():
    builder = ComputeGraphBuilder()
    partition_name = "p0"
    input0 = builder.input("input0")
    with builder.partition(partition_name):
        matmul0 = builder.matmul("matmul0", input0, input0)
        matmul1 = builder.matmul("matmul1", input0, input0)
    output0 = builder.output("output0", matmul0)
    output1 = builder.output("output1", matmul1)

    graph = builder.build()
    
    # Create pipeline
    pipeline = ComputePipeline(graph)

    # Enqueue input
    input_tensor = random_2d_tensor((10, 10))
    inputs = {}
    inputs[input0.name] = input_tensor
    pipeline.enqueue_input(PipelineInput(
        correlation_id="1",
        inputs=inputs
    ))

    # do some work
    work = pipeline.get_partition_work(partition_name)
    assert work is not None
    assert work.partition == partition_name
    assert len(work.graph.list_nodes()) == 2  # Fixed: We have two matmul operations
    assert len(work.graph.get_edges()) == 0  # No edges since they're internal to the partition
    assert isinstance(work.graph.get_node(matmul0.name), MatmulNode)
    assert isinstance(work.graph.get_node(matmul1.name), MatmulNode)
    result = simulate(work, llama_1b_cache(), False)
    pipeline.submit_partition_work(result)

    # Get output
    output = pipeline.dequeue_output(blocking=False)
    assert output is not None
    assert output.correlation_id == "1"
    assert output.outputs[output0.name].shape == input_tensor.shape
    assert output.outputs[output1.name].shape == input_tensor.shape

def test_flow():
    builder = ComputeGraphBuilder()
    input0 = builder.input("input0")
    with builder.partition("p0"):
        matmul0 = builder.matmul("matmul0", input0, input0)
        matmul1 = builder.matmul("matmul1", input0, input0)
    with builder.partition("p1"):
        matmul2 = builder.matmul("matmul2", matmul0, matmul1)
    output2 = builder.output("output2", matmul2)

    graph = builder.build()
    
    # Create pipeline
    pipeline = ComputePipeline(graph)

    # Enqueue input
    input_tensor = random_2d_tensor((10, 10))
    inputs = {}
    inputs[input0.name] = input_tensor
    pipeline.enqueue_input(PipelineInput(
        correlation_id="1",
        inputs=inputs
    ))

    # do some work
    work = pipeline.get_partition_work("p0")
    assert work is not None
    result = simulate(work, llama_1b_cache(), False)
    pipeline.submit_partition_work(result)

    work = pipeline.get_partition_work("p1")
    assert work is not None
    result = simulate(work, llama_1b_cache(), False)
    pipeline.submit_partition_work(result)

    # Get output
    output = pipeline.dequeue_output(blocking=False)
    assert output is not None
    assert output.correlation_id == "1"
    assert output.outputs[output2.name].shape == [10, 10]

def test_repeats():
    builder = ComputeGraphBuilder()
    partition_name = "p0"
    input0 = builder.input("input0")
    with builder.partition(partition_name):
        matmul0 = builder.matmul("matmul0", input0, input0)
        matmul1 = builder.matmul("matmul1", input0, input0)
    output0 = builder.output("output0", matmul0)
    output1 = builder.output("output1", matmul1)

    graph = builder.build()
    
    # Create pipeline
    pipeline = ComputePipeline(graph)

    # Enqueue input
    input_tensor = random_2d_tensor((10, 10))
    inputs = {}
    inputs[input0.name] = input_tensor
    for i in range(10):
        pipeline.enqueue_input(PipelineInput(
            correlation_id=f"{i}",
            inputs=inputs
        ))

    # do some work
    for i in range(10):
        work = pipeline.get_partition_work(partition_name)
        assert work is not None
        result = simulate(work, llama_1b_cache(), False)
        pipeline.submit_partition_work(result)

    assert pipeline.get_partition_work(partition_name) is None

    # Get output
    for i in range(10):
        output = pipeline.dequeue_output(blocking=False)
        assert output is not None
        assert output.correlation_id == f"{i}"
        assert output.outputs[output0.name].shape == input_tensor.shape
        assert output.outputs[output1.name].shape == input_tensor.shape

    assert pipeline.dequeue_output(blocking=False) is None
