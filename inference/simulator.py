import torch
from collections import defaultdict

from .tensor import Tensor
from .graph import (
    EdgeEncoding, NodeName, MatmulNode, InputNode, OutputNode, DEFAULT_NODE_OUTPUT,
    NodeInput, NodeOutput, SliceNode, UnsqueezeNode, BroadcastNode, CatNode,
    FixedNode, HadamardNode, AddNode, SoftmaxNode
)
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

    for n, _ in graph.nodes.items():
        print(n)
    for e in graph.edges:
        print(e.src, "=>", e.dst, e.dst_input)

    # Get the inputs
    input_table: dict[tuple[NodeName, NodeInput], torch.Tensor] = {}
    for input in work.inputs:
        torch_tensor = input.tensor.to_torch()
        input_table[(input.node, input.input)] = torch_tensor

    # Convert edges to forwards & backwards tables
    # All edges attached to an input
    forward_edges: dict[NodeName, dict[NodeInput, EdgeEncoding]] = {}
    # All edges attached to an output
    backward_edges: dict[NodeName, dict[NodeOutput, EdgeEncoding]] = {}
    for edge in graph.edges:
        if edge.dst not in forward_edges:
            forward_edges[edge.dst] = {}
        forward_edges[edge.dst][edge.dst_input] = edge

        if edge.src not in backward_edges:
            backward_edges[edge.src] = {}
        backward_edges[edge.src][edge.src_output] = edge

    # Find all nodes that need to be evaluated
    # This includes all destination nodes of inputs, plus any of their downstream nodes
    eval_nodes: list[NodeName] = []
    for input in work.inputs:
        eval_nodes.append(input.node)
    
    # Also include all nodes with no forward edges (outputs)
    output_nodes: list[NodeName] = []
    for node in graph.nodes.keys():
        if node not in backward_edges:
            output_nodes.append(node)

    # Recursively evaluate nodes with output nodes as entry points
    output_table: dict[(NodeName, NodeOutput), torch.Tensor] = {}
    def evaluate_output(node: NodeName, output: NodeOutput) -> torch.Tensor:
        # Helper to resolve an input
        def resolve_input(node: NodeName, input: NodeInput) -> torch.Tensor:
            if (node, input) in input_table:
                return input_table[(node, input)]
            else:
                # Find the edge that feeds the input
                edge = forward_edges[node][input]
                return evaluate_output(edge.src, edge.src_output)

        # If we've already evaluated this node and output, return the cached result
        if (node, output) in output_table:
            return output_table[(node, output)]

        encoded_node = graph.nodes[node]
        try:
            if encoded_node.type == "constant":
                raise NotImplementedError("Constants are not implemented")
            elif encoded_node.type == "matmul":
                lhs = resolve_input(node, MatmulNode.LHS)
                rhs = resolve_input(node, MatmulNode.RHS)
                output = lhs @ rhs
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "slice":
                input_tensor = resolve_input(node, SliceNode.INPUT)
                dim = encoded_node.dim
                start = encoded_node.start
                end = encoded_node.end
                output = input_tensor.narrow(dim, start, end - start)
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "unsqueeze":
                input_tensor = resolve_input(node, UnsqueezeNode.INPUT)
                dim = encoded_node.dim
                output = input_tensor.unsqueeze(dim)
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "broadcast":
                input_tensor = resolve_input(node, BroadcastNode.INPUT)
                dim = encoded_node.dim
                n = encoded_node.n
                output = input_tensor.expand(*[n if i == dim else -1 for i in range(input_tensor.dim())])
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "cat":
                a = resolve_input(node, CatNode.A)
                b = resolve_input(node, CatNode.B)
                dim = encoded_node.dim
                output = torch.cat([a, b], dim=dim)
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "fixed":
                output = encoded_node.tensor.to_torch()
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "hadamard":
                a = resolve_input(node, HadamardNode.A)
                b = resolve_input(node, HadamardNode.B)
                output = a * b
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "softmax":
                input_tensor = resolve_input(node, SoftmaxNode.INPUT) 
                dim = encoded_node.dim
                output = torch.softmax(input_tensor, dim=dim)
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
            elif encoded_node.type == "add":
                a = resolve_input(node, AddNode.A)
                b = resolve_input(node, AddNode.B)
                output = a + b
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            else:
                raise ValueError(f"Unknown node type: {encoded_node.type}")
        except Exception as e:
            print(f"Error evaluating {node} {output}: {e}")
            raise e

    # Evaluate all output nodes
    print("Output nodes:", output_nodes)
    for node in output_nodes:
        evaluate_output(node, DEFAULT_NODE_OUTPUT)

    # Build result
    output_assignments: list[OutputAssignment] = []
    for node in output_nodes:
        output_assignments.append(OutputAssignment(
            node=node,
            output=DEFAULT_NODE_OUTPUT,
                tensor=Tensor.from_torch(output_table[(node, DEFAULT_NODE_OUTPUT)])
            ))
    
    result = PartitionWorkResult(
        correlation_id=work.correlation_id,
        partition=work.partition,
        outputs=output_assignments
    )
    return result
