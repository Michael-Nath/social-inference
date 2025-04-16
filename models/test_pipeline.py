from .graph import ComputeGraph
from .pipeline import ComputePipeline
from .simulator import simulate
from .test_util import random_2d_tensor, llama_1b_cache

def test_simple_pipeline():
    graph = ComputeGraph()
    partition_name = "p0"
    input0 = graph.input("input0")
    with graph.partition(partition_name):
        matmul0 = graph.matmul("matmul0", input0, input0)
    output0 = graph.output("output0", matmul0)

    graph.freeze()
    
    # Create pipeline
    pipeline = ComputePipeline(graph)

    # Enqueue input
    input_tensor = random_2d_tensor("1", (10, 10))
    pipeline.enqueue_input({
        input0.name: input_tensor
    })

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
    output_tensor = pipeline.dequeue_output(blocking=False)
    assert output_tensor is not None
    assert output_tensor[output0.name].correlation_id == input_tensor.correlation_id
    # assert np.allclose(output_tensor[output0.name].tensor.elements, input_tensor.tensor.elements)
