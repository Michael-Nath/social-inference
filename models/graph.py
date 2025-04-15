from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

type NodeName = str
type PartitionName = str

PARTITION_INPUT: PartitionName = "__input__"
PARTITION_OUTPUT: PartitionName = "__output__"

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
        return {"output"}

class OutputNode(ComputeGraphNode):
    def __init__(self, name: NodeName, partition: PartitionName):
        super().__init__(name, partition)

    def get_input_names(self) -> set[str]:
        return {"input"}

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
        return {"output"}

class MatmulNode(ComputeGraphNode):
    def __init__(self, name: NodeName, partition: PartitionName):
        super().__init__(name, partition)

    def get_input_names(self) -> set[str]:
        return {"lhs", "rhs"}

    def get_output_names(self) -> set[str]:
        return {"output"}

def _check_partition_name(name: PartitionName):
    if name == PARTITION_INPUT or name == PARTITION_OUTPUT:
        raise ValueError(f"Invalid partition name: {name}")

@dataclass(eq=True, frozen=True)
class ComputeGraphEdge:
    """
    An edge is a 4-tuple (src, src_output, dst, dst_input)
    
    You will notice that "src_output" is always output. This is because, right
    now, we do not really support multiple outputs from a node.
    """
    src: NodeName
    src_output: str
    dst: NodeName
    dst_input: str
    
    def __repr__(self) -> str:
        return f"ComputeGraphEdge(src={self.src}, src_output={self.src_output}, dst={self.dst}, dst_input={self.dst_input})"
    def __str__(self) -> str:
        return f"{self.src}.{self.src_output} -> {self.dst}.{self.dst_input}"

class ComputeGraph:
    """
    A compute graph is an execution plan
    """

    nodes: dict[NodeName, ComputeGraphNode]
    """
    Nodes keyed by name
    """

    partitions: defaultdict[PartitionName, set[NodeName]]
    """
    Partitions keyed by name
    """

    _forward_edges: defaultdict[NodeName, list[ComputeGraphEdge]]
    """
    Edges keyed by source node
    """

    _backward_edges: defaultdict[NodeName, list[ComputeGraphEdge]]
    """
    Edges keyed by destination node
    """

    _active_partition: PartitionName | None

    def __init__(self):
        """
        Construct an empty compute graph
        """
        self.nodes = {}
        self._active_partition = None
        self.partitions = defaultdict(lambda: set())
        self._forward_edges = defaultdict(lambda: [])
        self._backward_edges = defaultdict(lambda: [])

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
        return f"ComputeGraph(nodes={self.nodes}, partitions={self.partitions})"
    
    def _make_edge(self, src: NodeName, src_output: str, dst: NodeName, dst_input: str):
        # Error check
        if src not in self.nodes:
            raise ValueError(f"Node {src} does not exist")
        if dst not in self.nodes:
            raise ValueError(f"Node {dst} does not exist")
        if src_output not in self.nodes[src].get_output_names():
            raise ValueError(f"Node {src} does not have output {src_output}")
        if dst_input not in self.nodes[dst].get_input_names():
            raise ValueError(f"Node {dst} does not have input {dst_input}")

        # Make the edge
        self._forward_edges[src].append(ComputeGraphEdge(src, src_output, dst, dst_input))
        self._backward_edges[dst].append(ComputeGraphEdge(src, src_output, dst, dst_input))
    
    def input(self, name: NodeName) -> InputNode:
        if name in self.nodes:
            raise ValueError(f"Node {name} already exists")

        self.nodes[name] = InputNode(name=name, partition=PARTITION_INPUT)
        self.partitions[PARTITION_INPUT].add(name)
        return self.nodes[name]
    
    def output(self, name: NodeName, x: ComputeGraphNode) -> OutputNode:
        if name in self.nodes:
            raise ValueError(f"Node {name} already exists")

        self.nodes[name] = OutputNode(name=name, partition=PARTITION_OUTPUT)
        self.partitions[PARTITION_OUTPUT].add(name)
        self._make_edge(x.name, "output", name, "input")
        return self.nodes[name]
    
    def constant(self, name: NodeName, tensor_name: str) -> ConstantNode:
        if name in self.nodes:
            raise ValueError(f"Node {name} already exists")
        if self._active_partition is None:
            raise ValueError(f"Node {name} must be created within a partition")

        self.nodes[name] = ConstantNode(name=name, partition=self._active_partition, tensor_name=tensor_name)
        self.partitions[self._active_partition].add(name)
        self._make_edge(name, "output", name, "input")
        return self.nodes[name]
    
    def matmul(self, name: NodeName, x: ComputeGraphNode, y: ComputeGraphNode) -> MatmulNode:
        if name in self.nodes:
            raise ValueError(f"Node {name} already exists")
        if self._active_partition is None:
            raise ValueError(f"Node {name} must be created within a partition")

        self.nodes[name] = MatmulNode(name=name, partition=self._active_partition)
        self.partitions[self._active_partition].add(name)
        self._make_edge(x.name, "output", name, "lhs")
        self._make_edge(y.name, "output", name, "rhs")

        return self.nodes[name]
    
    def extract_partition(self, name: PartitionName) -> "ComputeGraph":
        """
        Extract a partition from the compute graph.

        By definition, the new graph will have exactly one partition named name
        containing the entire graph.
        """
        new_graph = ComputeGraph()
        for node_name in self.partitions[name]:
            new_graph.nodes[node_name] = self.nodes[node_name]
            new_graph.partitions[name].add(node_name)
            # Copy forward and backward edges that are within the partition
            for edge in self._forward_edges[node_name]:
                if edge.src in self.partitions[name] and edge.dst in self.partitions[name]:
                    new_graph._forward_edges[edge.src].append(edge)
            for edge in self._backward_edges[node_name]:
                if edge.src in self.partitions[name] and edge.dst in self.partitions[name]:
                    new_graph._backward_edges[edge.dst].append(edge)
        return new_graph
    
    def identify_forward_cuts(self, partition: PartitionName) -> set[ComputeGraphEdge]:
        """
        Returns a list of forward edges that are cut by a partition.
        """
        result = set()
        for dst, edges in self._backward_edges.items():
            if dst not in self.partitions[partition]:
                continue
            for edge in edges:
                if edge.src not in self.partitions[partition]:
                    result.add(edge)
        return result

    def identify_backward_cuts(self, partition: PartitionName) -> set[ComputeGraphEdge]:
        """
        Returns a list of backward edges that are cut by a partition.
        """
        result = set()
        for src, edges in self._forward_edges.items():
            if src not in self.partitions[partition]:
                continue
            for edge in edges:
                if edge.dst not in self.partitions[partition]:
                    result.add(edge)
        return result

class ComputePipeline:
    """
    A compute pipeline stores the execution state of a compute graph
    """

    pass