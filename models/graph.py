from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Annotated, Literal, Union
from copy import deepcopy
from pydantic import BaseModel, Field

type NodeName = str
type NodeInput = str
type NodeOutput = str
type PartitionName = str

PARTITION_INPUT: PartitionName = "__input__"
PARTITION_OUTPUT: PartitionName = "__output__"

DEFAULT_NODE_OUTPUT: NodeOutput = "output"
"""
Our logic mostly supports multiple outputs from a node, but not fully. Use this.
"""

class ConstantNodeEncoding(BaseModel):
    """
    API-encoded constant node.
    """
    type: Literal["constant"]
    name: NodeName
    tensor_name: str

class MatmulNodeEncoding(BaseModel):
    """
    API-encoded matmul node.
    """
    type: Literal["matmul"]
    name: NodeName

type NodeEncoding = Annotated[Union[MatmulNodeEncoding, ConstantNodeEncoding], Field(discriminator="type")]
"""
API-encoded node.

No Input/Output nodes since those are never sent to workers.
"""

class EdgeEncoding(BaseModel):
    """
    API-encoded edge.
    """
    src: NodeName
    src_output: NodeOutput
    dst: NodeName
    dst_input: NodeInput

class GraphEncoding(BaseModel):
    """
    API-encoded graph.
    """
    nodes: dict[NodeName, NodeEncoding]
    edges: list[EdgeEncoding]

class ComputeGraphNode(ABC):
    """
    A node in the compute graph. Represents a single operation.

    Graph relations are stored in ComputeGraph. Nodes hold no references to each
    other. Instead, nodes hold relevant information for their operation e.g.
    safe tensor name for constant operation.

    The way nodes *do* participate in the graph is by indicating valid input and output names.
    """

    name: NodeName
    partition: PartitionName
    
    def __init__(self, name: NodeName, partition: PartitionName):
        self.name = name
        self.partition = partition

    @abstractmethod
    def get_input_names(self) -> set[str]:
        """
        Get the names of the inputs to this node.
        """
        pass

    @abstractmethod
    def get_output_names(self) -> set[str]:
        """
        Get the names of the outputs from this node.
        """
        pass

class InputNode(ComputeGraphNode):
    def __init__(self, name: NodeName, partition: PartitionName):
        super().__init__(name, partition)

    def get_input_names(self) -> set[str]:
        return set()

    def get_output_names(self) -> set[str]:
        return {DEFAULT_NODE_OUTPUT}

class OutputNode(ComputeGraphNode):
    INPUT: NodeInput = "input"

    def __init__(self, name: NodeName, partition: PartitionName):
        super().__init__(name, partition)

    def get_input_names(self) -> set[str]:
        return {self.INPUT}

    def get_output_names(self) -> set[str]:
        return set()

class ConstantNode(ComputeGraphNode):
    tensor_name: str
    """
    The name of the safe tensor.
    """


    def __init__(self, name: NodeName, partition: PartitionName, tensor_name: str):
        super().__init__(name, partition)
        self.tensor_name = tensor_name

    def get_input_names(self) -> set[str]:
        return set()

    def get_output_names(self) -> set[str]:
        return {DEFAULT_NODE_OUTPUT}

class MatmulNode(ComputeGraphNode):
    LHS: NodeInput = "lhs"
    RHS: NodeInput = "rhs"

    def __init__(self, name: NodeName, partition: PartitionName):
        super().__init__(name, partition)

    def get_input_names(self) -> set[str]:
        return {self.LHS, self.RHS}

    def get_output_names(self) -> set[str]:
        return {DEFAULT_NODE_OUTPUT}
    
def _check_partition_name(name: PartitionName):
    if name == PARTITION_INPUT or name == PARTITION_OUTPUT:
        raise ValueError(f"Invalid partition name: {name}")

@dataclass(eq=True, frozen=True)
class ComputeGraphEdge:
    """
    An edge is a 4-tuple (src, src_output, dst, dst_input)
    
    You will notice that "src_output" is always output. This is because, right
    now, we do not really support multiple outputs from a node.

    This type is hashable.
    """
    src: NodeName
    src_output: NodeOutput
    dst: NodeName
    dst_input: NodeInput
    
    def __repr__(self) -> str:
        return f"ComputeGraphEdge(src={self.src}, src_output={self.src_output}, dst={self.dst}, dst_input={self.dst_input})"
    def __str__(self) -> str:
        return f"{self.src}.{self.src_output} -> {self.dst}.{self.dst_input}"

class ComputeGraphBuilder:
    """
    Builder for a compute graph.
    """

    _nodes: dict[NodeName, ComputeGraphNode]
    _edges: list[ComputeGraphEdge]

    _active_partition: PartitionName | None

    def __init__(self):
        self._nodes = {}
        self._edges = []
        self._active_partition = None

    @contextmanager
    def partition(self, name: PartitionName):
        """
        Helper context manager to make it easier to create many nodes with the
        same partition.
        """
        try:
            if name == PARTITION_INPUT or name == PARTITION_OUTPUT:
                raise ValueError(f"Invalid partition name: {name}")
            self._active_partition = name
            yield
        finally:
            self._active_partition = None   

    def _make_edge(self, src: NodeName, src_output: NodeOutput, dst: NodeName, dst_input: NodeInput):
        # Error check
        if src not in self._nodes:
            raise ValueError(f"Node {src} does not exist")
        if dst not in self._nodes:
            raise ValueError(f"Node {dst} does not exist")
        if src_output not in self._nodes[src].get_output_names():
            raise ValueError(f"Node {src} does not have output {src_output}")
        if dst_input not in self._nodes[dst].get_input_names():
            raise ValueError(f"Node {dst} does not have input {dst_input}")

        # Make the edge
        self._edges.append(ComputeGraphEdge(src, src_output, dst, dst_input))

    def input(self, name: NodeName) -> InputNode:
        if name in self._nodes:
            raise ValueError(f"Node {name} already exists")

        self._nodes[name] = InputNode(name=name, partition=PARTITION_INPUT)
        return self._nodes[name]
    
    def output(self, name: NodeName, x: ComputeGraphNode) -> OutputNode:
        if name in self._nodes:
            raise ValueError(f"Node {name} already exists")
        self._nodes[name] = OutputNode(name=name, partition=PARTITION_OUTPUT)
        self._make_edge(x.name, DEFAULT_NODE_OUTPUT, name, OutputNode.INPUT)
        return self._nodes[name]

    def constant(self, name: NodeName, tensor_name: str) -> ConstantNode:
        if name in self._nodes:
            raise ValueError(f"Node {name} already exists")
        if self._active_partition is None:
            raise ValueError(f"Node {name} must be created within a partition")

        self._nodes[name] = ConstantNode(name=name, partition=self._active_partition, tensor_name=tensor_name)
        self._make_edge(name, DEFAULT_NODE_OUTPUT, name, InputNode.INPUT)
        return self._nodes[name]
    
    def matmul(self, name: NodeName, lhs: ComputeGraphNode, rhs: ComputeGraphNode) -> MatmulNode:
        if name in self._nodes:
            raise ValueError(f"Node {name} already exists")
        if self._active_partition is None:
            raise ValueError(f"Node {name} must be created within a partition")

        self._nodes[name] = MatmulNode(name=name, partition=self._active_partition)
        self._make_edge(lhs.name, DEFAULT_NODE_OUTPUT, name, MatmulNode.LHS)
        self._make_edge(rhs.name, DEFAULT_NODE_OUTPUT, name, MatmulNode.RHS)
        return self._nodes[name]

    def build(self, copy: bool = False) -> ComputeGraph:
        """
        Build the compute graph.
        """
        return ComputeGraph(self._nodes.values(), self._edges, copy=copy)

class ComputeGraph:
    """
    A compute graph is an execution plan
    """

    _nodes: dict[NodeName, ComputeGraphNode]
    """
    Nodes keyed by name
    """

    _partitions: defaultdict[PartitionName, set[NodeName]]
    """
    Partitions keyed by name
    """

    _forward_edges: defaultdict[NodeName, set[ComputeGraphEdge]]
    """
    Edges keyed by source node
    """

    _backward_edges: defaultdict[NodeName, set[ComputeGraphEdge]]
    """
    Edges keyed by destination node
    """

    _cached_forward_cuts: dict[PartitionName, set[ComputeGraphEdge]]
    """
    Cached forward cuts. Only populated after the graph has been frozen.
    """

    _cached_backward_cuts: dict[PartitionName, set[ComputeGraphEdge]]
    """
    Cached backward cuts. Only populated after the graph has been frozen.
    """

    def __init__(self, nodes: list[ComputeGraphNode], edges: list[ComputeGraphEdge], copy: bool = False):
        """
        Construct a compute graph from a list of nodes and edges.

        The initializer does *no* error checking, and will create nodes and edges as-given.

        You should use ComputeGraphBuilder instead.

        Args:
            nodes: List of nodes to add to the graph
            edges: List of edges to add to the graph
        """

        self._nodes = {node.name: deepcopy(node) if copy else node for node in nodes}
        self._partitions = defaultdict(lambda: set())
        for node in nodes:
            self._partitions[node.partition].add(node.name)

        self._forward_edges = defaultdict(lambda: set())
        self._backward_edges = defaultdict(lambda: set())
        for edge in (deepcopy(edges) if copy else edges):
            self._forward_edges[edge.src].add(edge)
            self._backward_edges[edge.dst].add(edge)

        self._cached_forward_cuts = {}
        self._cached_backward_cuts = {}

    @contextmanager
    def partition(self, name: PartitionName):
        """
        Helper context manager to make it easier to create many nodes with the
        same partition.
        """
        try:
            _check_partition_name(name)
            self._active_partition = name
            yield
        finally:
            self._active_partition = None

    def __repr__(self) -> str:
        return f"ComputeGraph(nodes=<{len(self._nodes)}>, partitions=<{len(self._partitions)}>, edges=<{len(self._forward_edges)}>)"
    
    def get_node(self, name: NodeName) -> ComputeGraphNode:
        """
        Get a node by name.
        """
        return self._nodes[name]
    
    def list_partition(self, partition: PartitionName) -> set[NodeName]:
        """
        Get all node names in a partition.
        """
        return self._partitions[partition]
    
    def list_nodes(self) -> set[NodeName]:
        """
        Get all node names in the graph.
        """
        return set(self._nodes.keys())
    
    def get_edges(self) -> set[ComputeGraphEdge]:
        """
        Get all edges in the graph.
        """
        # Combine all edge sets into a single set
        all_edges = set()
        for edge_set in self._forward_edges.values():
            all_edges.update(edge_set)
        return all_edges

    def get_partition(self, node: NodeName) -> PartitionName:
        """
        Get the partition for a given node.
        """
        return self._nodes[node].partition

    def get_partitions(self) -> set[PartitionName]:
        """
        Get the list of partitions in the graph.
        """
        return set(self._partitions.keys())

    def get_forward_edges(self, src: NodeName, src_output: NodeOutput | None = None) -> set[ComputeGraphEdge]:
        """
        Get the forward edges for a given node and output. In other words, the
        inputs this output flows into.

        Args:
            src: The source node
            src_output: Optional output from the source node. If not provided, all forward edges for the source node are returned.

        Returns:
            The forward edges for the given node and output.
        """
        if src_output is None:
            return self._forward_edges[src]
        else:
            return {e for e in self._forward_edges[src] if e.src_output == src_output}
        
    def get_backward_edges(self, dst: NodeName, dst_input: NodeInput | None = None) -> set[ComputeGraphEdge]:
        """
        Get the backward edges for a given node and input. In other words, the
        outputs that flow into this input.

        Args:
            dst: The destination node
            dst_input: Optional input to the destination node. If not provided, all backward edges for the destination node are returned.

        Returns:
            The backward edges for the given node and input.
        """
        if dst_input is None:
            return self._backward_edges[dst]
        else:
            return {e for e in self._backward_edges[dst] if e.dst_input == dst_input}
    
    def extract_partition(self, name: PartitionName, include_cut_edges: bool = False, copy: bool = False) -> "ComputeGraph":
        """
        Extract a frozen partition from the compute graph.

        By definition, the new graph will have exactly one partition named name
        containing the entire graph.

        Args:
            partition: The name of the partition to extract
        Keyword Args:
            include_cut_edges: If True (default False), the new graph will include the edges
                that were cut by the partition. This is useful for execution, 
                since we assign values per-edge and input/output edges will be cut.
            copy: If True (default False), deepcopy nodes. Otherwise, creates a view.
        """
        partition_node_names = self._partitions[name]
        partition_nodes = [self._nodes[name] for name in partition_node_names]

        # Extract edges
        partition_edges = []
        for edges in self._forward_edges.values():
            for edge in edges:
                in_both = edge.src in partition_node_names and edge.dst in partition_node_names
                in_one = edge.src in partition_node_names or edge.dst in partition_node_names
                include_edge = in_both or (include_cut_edges and in_one)
                if include_edge:
                    partition_edges.append(edge)

        # Do not copy since we are creating an immutable view
        new_graph = ComputeGraph(partition_nodes, partition_edges)
        return new_graph
    
    def identify_forward_cuts(self, partition: PartitionName) -> set[ComputeGraphEdge]:
        """
        Returns a list of forward edges (edges flowing *into* the partition) that are cut by a partition.
        """
        if partition in self._cached_forward_cuts:
            return self._cached_forward_cuts[partition]

        result = set()
        for dst, edges in self._backward_edges.items():
            if dst not in self._partitions[partition]:
                continue
            for edge in edges:
                if edge.src not in self._partitions[partition]:
                    result.add(edge)

        self._cached_forward_cuts[partition] = result

        return result

    def identify_backward_cuts(self, partition: PartitionName) -> set[ComputeGraphEdge]:
        """
        Returns a list of backward edges (edges flowing *out of* the partition) that are cut by a partition.
        """
        if partition in self._cached_backward_cuts:
            return self._cached_backward_cuts[partition]

        result = set()
        for src, edges in self._forward_edges.items():
            if src not in self._partitions[partition]:
                continue
            for edge in edges:
                if edge.dst not in self._partitions[partition]:
                    result.add(edge)

        self._cached_backward_cuts[partition] = result

        return result
    
    def encode(self) -> GraphEncoding:
        """
        Encode the graph into API format.
        """
        nodes: dict[NodeName, NodeEncoding] = {}
        for node_name, node in self._nodes.items():
            if isinstance(node, MatmulNode):
                nodes[node_name] = MatmulNodeEncoding(type="matmul", name=node_name)
            elif isinstance(node, ConstantNode):
                nodes[node_name] = ConstantNodeEncoding(type="constant", name=node_name, tensor_name=node.tensor_name)
        edges: list[EdgeEncoding] = []
        for e in self._forward_edges.values():
            for edge in e:
                edges.append(EdgeEncoding(src=edge.src, src_output=edge.src_output, dst=edge.dst, dst_input=edge.dst_input))
        return GraphEncoding(nodes=nodes, edges=edges)
