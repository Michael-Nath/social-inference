import numpy as np
from collections import defaultdict

from .tensor import Tensor
from .graph import EdgeEncoding, NodeName
from .pipeline import OutputAssignment, PartitionWork, PartitionWorkResult
from .cache import SafeTensorCache

def simulate(work: PartitionWork, tensor_cache: SafeTensorCache) -> PartitionWorkResult:
    """
    Simulates a worker.

    Args:
        work: The work to simulate
        tensor_cache: The tensor cache to use for constant tensors
    """
    # Get the graph
    graph = work.graph

    # Get the inputs
    input_table: dict[tuple[NodeName, str], np.ndarray] = {}
    for input in work.inputs:
        np_tensor = input.tensor.to_numpy()
        input_table[(input.node, input.input)] = np_tensor

    # Convert edges to forwards & backwards tables
    forward_edges: dict[NodeName, dict[str, EdgeEncoding]] = {}
    backward_edges: dict[NodeName, dict[str, EdgeEncoding]] = {}
    for edge in graph.edges:
        forward_edges[edge.src] = {}
        forward_edges[edge.src][edge.dst_input] = edge
        backward_edges[edge.dst] = {}
        backward_edges[edge.dst][edge.src_output] = edge

    # Collect output nodes
    output_nodes: list[NodeName] = []
    for node in graph.nodes.keys():
        if node not in forward_edges:
            output_nodes.append(node)

    # Recursively evaluate nodes with output nodes as entry points
    output_table: dict[(NodeName, str), np.ndarray] = {}
    def evaluate_output(node: NodeName, output: str) -> np.ndarray:
        # Helper to resolve an input
        def resolve_input(node: NodeName, input: str) -> np.ndarray:
            if (node, input) in input_table:
                return input_table[(node, input)]
            else:
                edge = backward_edges[node][input]
                return evaluate_output(edge.src, edge.src_output)

        encoded_node = graph.nodes[node]
        if encoded_node.type == "constant":
            raise NotImplementedError("Constants are not implemented")
        elif encoded_node.type == "matmul":
            lhs = resolve_input(node, "lhs")
            rhs = resolve_input(node, "rhs")
            output = lhs @ rhs
            output_table[(node, "output")] = output
            return output
        else:
            raise ValueError(f"Unknown node type: {encoded_node.type}")

    for node in output_nodes:
        evaluate_output(node, "output")

    # Build result
    output_assignments: list[OutputAssignment] = []
    for node in output_nodes:
        output_assignments.append(OutputAssignment(
            node=node,
            output="output",
            tensor=Tensor.from_numpy(output_table[(node, "output")])
        ))

    # print("Input table:")
    # for key, value in input_table.items():
    #     print(f"{key}: {value}")
    # print("Edges:")
    # for edge in graph.edges:
    #     print(f"{edge.src} ({edge.src_output}) -> {edge.dst} ({edge.dst_input})")
    # print("Output nodes:")
    # print(output_nodes)
    # print("Output table:")
    # for key, value in output_table.items():
    #     print(f"{key}: {value}")

    result = PartitionWorkResult(
        correlation_id=work.correlation_id,
        partition=work.partition,
        outputs=output_assignments
    )
    return result
