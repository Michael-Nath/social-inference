import torch
from collections import defaultdict

from .tensor import Tensor
from .graph import (
    EdgeEncoding, NodeName, MatmulNode, InputNode, OutputNode, DEFAULT_NODE_OUTPUT,
    NodeInput, NodeOutput, SliceNode, UnsqueezeNode, BroadcastNode, CatNode,
    FixedNode, HadamardNode, AddNode, IndexNode, ShapeNode, SoftmaxNode, ReshapeNode, TransposeNode, DivideNode
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
            
        def check_dim(dim: int):
            if dim < 0:
                raise NotImplementedError("Negative (inferred) dimensions not supported")
            return dim
            
        def check_shapes(*args, except_dim: int | None = None):
            """
            Checks that all shapes are equal (except except_dim if that is set)
            """
            if except_dim is None:
                # Check all dimensions match exactly
                for t in args:
                    if t.shape != args[0].shape:
                        raise ValueError(f"Shape {t.shape} does not match shape {args[0].shape}")
            else:
                # Check all dimensions except the specified one
                for t in args:
                    if len(t.shape) != len(args[0].shape):
                        raise ValueError(f"Tensor rank {len(t.shape)} does not match {len(args[0].shape)}")
                    
                    for i, (s1, s2) in enumerate(zip(t.shape, args[0].shape)):
                        if i != except_dim and s1 != s2:
                            raise ValueError(f"Shape {t.shape} does not match shape {args[0].shape} (mismatch at dimension {i})")

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
                # check_shapes(lhs, rhs)

                output = lhs @ rhs
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "slice":
                input_tensor = resolve_input(node, SliceNode.INPUT)
                dim = check_dim(encoded_node.dim)
                start = check_dim(encoded_node.start)
                end = check_dim(encoded_node.end)

                output = input_tensor.narrow(dim, start, end - start)
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "unsqueeze":
                input_tensor = resolve_input(node, UnsqueezeNode.INPUT)
                dim = check_dim(encoded_node.dim)

                output = input_tensor.unsqueeze(dim)
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "broadcast":
                input_tensor = resolve_input(node, BroadcastNode.INPUT)
                dim = check_dim(encoded_node.dim)
                n = check_dim(encoded_node.n) # n is a new dimension size

                output = input_tensor.expand(*[n if i == dim else -1 for i in range(input_tensor.dim())])
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "cat":
                a = resolve_input(node, CatNode.A)
                b = resolve_input(node, CatNode.B)
                dim = check_dim(encoded_node.dim)
                check_shapes(a, b, except_dim=dim)
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
                check_shapes(a,b)

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
                check_shapes(a,b)
                output = a + b
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "divide":
                a = resolve_input(node, DivideNode.A)
                b = resolve_input(node, DivideNode.B)
                check_shapes(a,b)
                output = a / b
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "index":
                input_tensor = resolve_input(node, IndexNode.INPUT)
                index_tensor = resolve_input(node, IndexNode.INDEX)

                input_tensor_shape = input_tensor.shape
                index_shape = index_tensor.shape
                if len(index_shape) != 1 or (index_shape[0] < 1 or index_shape[0] > len(input_tensor_shape)):
                    raise ValueError(f"Invalid index {index_tensor} for input shape {input_tensor_shape}")

                # Convert index tensor to list of integers
                index_list = index_tensor.tolist()
                if not all(isinstance(idx, int) for idx in index_list):
                    raise ValueError(f"Index tensor must contain integers, got {index_list}")
                
                # Validate that all indices are non-negative and less than the corresponding dimension size
                for i, idx in enumerate(index_list):
                    if idx < 0:
                        raise ValueError(f"Index must be non-negative, got {idx} at position {i}")
                    if i < len(input_tensor_shape) and idx >= input_tensor_shape[i]:
                        raise ValueError(f"Index {idx} at position {i} is out of bounds for dimension size {input_tensor_shape[i]}")

                # To preserve shape: wrap each index as a slice if it’s the last one
                indexing = []
                for dim, idx in enumerate(index_list):
                    if dim == len(index_list) - 1:
                        indexing.append(slice(idx, idx + 1))  # turn scalar index into slice
                    else:
                        indexing.append(idx)
                
                output = input_tensor[tuple(indexing)]
                print(f"{input_tensor}[{index_tensor}] => {output}")
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "shape":
                input_tensor = resolve_input(node, ShapeNode.INPUT)
                output = torch.tensor(input_tensor.shape, dtype=torch.long)
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "reshape":
                input_tensor = resolve_input(node, ReshapeNode.INPUT)
                shape = resolve_input(node, ReshapeNode.DIMS).int()
                output = torch.reshape(input_tensor, tuple(shape.tolist()))
                output_table[((node, DEFAULT_NODE_OUTPUT))] = output
                return output
            elif encoded_node.type == "transpose":
                input_tensor = resolve_input(node, TransposeNode.INPUT)
                dim0 = encoded_node.dim0
                dim1 = encoded_node.dim1

                output = torch.transpose(input_tensor, dim0, dim1)
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            else:
                raise ValueError(f"Unknown node type: {encoded_node.type}")
        except Exception as e:
            print(f"Error evaluating {node} {output}: {e}")
            raise e

    # Evaluate all output nodes
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
