import torch
from collections import defaultdict

from .tensor import Tensor
from .graph import (
    EdgeEncoding, NodeName, MatmulNode, DEFAULT_NODE_OUTPUT,
    NodeInput, NodeOutput, SliceNode, UnsqueezeNode, BroadcastNode, CatNode,
    HadamardNode, AddNode, IndexNode, ShapeNode, SoftmaxNode, DivNode,
    FloorNode, CeilNode, ReshapeNode, TransposeNode, DebugNode
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
        def check_shape(tensor: torch.Tensor, shape: list[int]):
            """
            Checks that a tensor matches the expected shape.
            If shape[i] = -1, it means "don't care" about that dimension.
            """
            if len(tensor.shape) != len(shape):
                raise ValueError(f"Tensor rank {len(tensor.shape)} does not match expected rank {len(shape)}")
            
            for i, (actual, expected) in enumerate(zip(tensor.shape, shape)):
                if expected != -1 and actual != expected:
                    raise ValueError(f"Tensor shape {tensor.shape} does not match expected shape {shape} (mismatch at dimension {i})")
            return tensor
        
        def check_shapes_match(*args, except_dim: int | None = None):
            """
            Checks that all shapes are equal (except except_dim if that is set)
            """
            if except_dim is None:
                # Check all dimensions match exactly
                for t in args:
                    check_shape(t, list(args[0].shape))
            else:
                # Check all dimensions except the specified one
                for t in args:
                    # Create expected shape with -1 for the except_dim
                    expected_shape = list(args[0].shape)
                    if except_dim < len(expected_shape):
                        expected_shape[except_dim] = -1
                    check_shape(t, expected_shape)

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
                # check_shapes_match(lhs, rhs)
                assert lhs.shape[-1] == rhs.shape[-2], f"matmul shape mismatch in {lhs.shape} @ {rhs.shape}"
                output = lhs @ rhs
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "slice":
                input_tensor = resolve_input(node, SliceNode.INPUT)
                dim = check_shape(resolve_input(node, SliceNode.DIM), [1]).item()
                start = check_shape(resolve_input(node, SliceNode.START), [1]).item()
                end = check_shape(resolve_input(node, SliceNode.END), [1]).item()

                output = input_tensor.narrow(dim, start, end - start)
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "unsqueeze":
                input_tensor = resolve_input(node, UnsqueezeNode.INPUT)
                # Resolve dim dynamically, ensure it's a 1-element tensor, and get the integer value
                dim_tensor = resolve_input(node, UnsqueezeNode.DIM)
                dim = check_shape(dim_tensor, [1]).item()

                output = input_tensor.unsqueeze(dim)
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "broadcast":
                input_tensor = resolve_input(node, BroadcastNode.INPUT)
                # Resolve dim dynamically
                dim_tensor = resolve_input(node, BroadcastNode.DIM)
                dim = check_shape(dim_tensor, [1]).item()
                # Resolve n dynamically
                n_tensor = resolve_input(node, BroadcastNode.N)
                n = check_shape(n_tensor, [1]).item() # n is the new dimension size

                output = input_tensor.expand(*[n if i == dim else -1 for i in range(input_tensor.dim())])
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "cat":
                a = resolve_input(node, CatNode.A)
                b = resolve_input(node, CatNode.B)
                # Resolve dim dynamically
                dim_tensor = resolve_input(node, CatNode.DIM)
                dim = check_shape(dim_tensor, [1]).item()
                check_shapes_match(a, b, except_dim=dim)
                
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
                # check_shapes_match(a,b)
                output = a * b
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "debug":
                tensor = resolve_input(node, DebugNode.INPUT)
                print(f"DEBUG<{node}>:")
                print("=================================")
                print(tensor)
                print("=================================")
                output_table[(node, DEFAULT_NODE_OUTPUT)] = tensor
                return tensor 
            elif encoded_node.type == "softmax":
                input_tensor = resolve_input(node, SoftmaxNode.INPUT) 
                # Resolve dim dynamically
                dim_tensor = resolve_input(node, SoftmaxNode.DIM)
                dim = check_shape(dim_tensor, [1]).item()
                output = torch.softmax(input_tensor, dim=dim)
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                # Need to return the output here
                return output
            elif encoded_node.type == "add":
                a = resolve_input(node, AddNode.A)
                b = resolve_input(node, AddNode.B)
                check_shapes_match(a,b)

                output = a + b
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "div":
                a = resolve_input(node, DivNode.A)
                b = resolve_input(node, DivNode.B)
                check_shapes_match(a,b)

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
                output_table[(((node, DEFAULT_NODE_OUTPUT)))] = output
                return output
            elif encoded_node.type == "floor":
                input_tensor = resolve_input(node, FloorNode.INPUT)
                output = torch.floor(input_tensor).to(torch.long)
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            elif encoded_node.type == "ceil":
                input_tensor = resolve_input(node, CeilNode.INPUT)
                output = torch.ceil(input_tensor).to(torch.long)
                output_table[(node, DEFAULT_NODE_OUTPUT)] = output
                return output
            else:
                raise ValueError(f"Unknown node type: {encoded_node.type}")
        except Exception as e:
            print(f"Error evaluating {node} {output}: {e}")
            # Print output table for debugging
            print(f"\nError occurred in node: {node}")
            print(f"Output table contents:")
            for (n, out), tensor in output_table.items():
                print(f"  {n}.{out}: {tensor.shape}")
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
