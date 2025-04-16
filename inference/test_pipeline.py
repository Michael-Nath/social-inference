from .graph import ComputeGraphBuilder, ComputeGraph
from .pipeline import ComputePipeline, PipelineInput, PipelineOutput
from .simulator import simulate
from .test_util import random_2d_tensor, llama_1b_cache

def test_simple_pipeline():
    builder = ComputeGraphBuilder()
    partition_name = "p0"
    input0 = builder.input("input0")
    with builder.partition(partition_name):
        matmul0 = builder.matmul("matmul0", input0, input0)
    output0 = builder.output("output0", matmul0)

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
    assert len(work.graph.nodes) == 1
    assert len(work.graph.edges) == 0 # No edges
    assert work.graph.nodes[matmul0.name].type == "matmul"
    result = simulate(work, llama_1b_cache())
    pipeline.submit_partition_work(result)

    # Get output
    output = pipeline.dequeue_output(blocking=False)
    assert output is not None
    assert output.correlation_id == "1"
    assert output.outputs[output0.name].shape == input_tensor.shape

# TODO: Many more tests of various weird behaviors