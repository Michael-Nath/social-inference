from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import threading
import time

from inference.encoding import read_be_int, read_encoded_string, size_encoded_string, write_be_int, write_encoded_string, write_encoded_bool, read_bool

from .graph import ComputeGraph, ComputeGraphEdge, NodeName, PartitionName, PARTITION_INPUT, PARTITION_OUTPUT
from .tensor import Tensor, read_encoded_tensor, size_encoded_tensor, write_encoded_tensor
from .queue import CorrelatedQueue, CorrelatedTensor

import torch

@dataclass
class PipelineInput:
    """
    Input to the pipeline
    """
    correlation_id: str
    inputs: dict[NodeName, Tensor]

@dataclass
class PipelineOutput:
    """
    Output from the pipeline
    """
    correlation_id: str
    outputs: dict[NodeName, Tensor]

@dataclass
class InputAssignment:
    node: NodeName
    input: str
    tensor: Tensor

@dataclass
class OutputAssignment:
    node: NodeName
    output: str
    tensor: Tensor


@dataclass
class PartitionWork:
    """
    Work to be done for a partition
    """
    correlation_id: str
    partition: PartitionName
    graph: ComputeGraph
    inputs: list[InputAssignment]
    should_trace: bool = False

@dataclass
class SingleStepChunk:
    correlation_id: str
    partition: PartitionName
    outputs: list[OutputAssignment]
    last_chunk: bool

    def consistent_with(self, other: PartitionWorkResult):
        if other.correlation_id != self.correlation_id:
            raise ValueError(f"Correlation {self.correlation_id} != {other.correlation_id}")
        if other.partition != self.partition:
            raise ValueError(f"Partition {self.partition} != {other.partition}")
        our_outputs = {}
        for our in self.outputs:
            our_outputs[(our.node, our.output)] = our
            other_node = None
            for _other_node in other.outputs:
                if _other_node.node == our.node and _other_node.output == our.output:
                    other_node = _other_node
                    break
            if other_node is None:
                breakpoint()
                raise ValueError(f"Extraneous output {our}")

            # Check if tensor dtypes match
            if our.tensor.dtype != other_node.tensor.dtype:
                breakpoint()
                raise ValueError(f"Output {other_node.node}.{other_node.output} dtype mismatch: {other_node.tensor.dtype} != {our.tensor.dtype}")
            if not torch.allclose(other_node.tensor.to_torch(), our.tensor.to_torch(), rtol=1e-3, atol=1e-4):
                breakpoint()
                raise ValueError(f"Output {other_node}={other.node.tensor.to_torch()} does not match our output {our}={our.tensor.to_torch()}")
    
    def encode(self, data: bytearray, offset: int):
        offset = write_encoded_string(data, offset, self.correlation_id)
        offset = write_encoded_string(data, offset, self.partition)
        num_outputs = len(self.outputs)
        offset = write_be_int(data, offset, num_outputs)
        for output in self.outputs:
            offset = write_encoded_output_assignment(data, offset, output)
        offset = write_encoded_bool(data, offset, self.last_chunk)
        return offset
    
    @classmethod
    def decode(cls, offset: int, data: bytes):
        
        c_id, offset = read_encoded_string(offset, data)
        partition, offset = read_encoded_string(offset, data)
        num_outputs, offset = read_be_int(offset, data)
        outputs = []
        for _ in range(num_outputs):
            output, offset = read_encoded_output_assignment(offset, data)
            outputs.append(output)
        last_chunk, offset = read_bool(offset, data)
        return cls(c_id, partition, outputs, last_chunk), offset
        
    def encoded_size(self):
        size = size_encoded_string(self.correlation_id)
        size += size_encoded_string(self.partition)
        for output in self.outputs:
            size += size_encoded_output_assignment(output)
        size += size_encoded_bool(self.last_chunk)
        return size

@dataclass
class PartitionWorkResult:
    """
    Result of partition work
    """
    correlation_id: str
    partition: PartitionName
    outputs: list[OutputAssignment]

    def close_to(self, other: PartitionWorkResult):
        if other.correlation_id != self.correlation_id:
            raise ValueError(f"Correlation {self.correlation_id} != {other.correlation_id}")
        if other.partition != self.partition:
            raise ValueError(f"Partition {self.partition} != {other.partition}")
        if len(self.outputs) != len(other.outputs):
            raise ValueError(f"Outputs {self.outputs} != {other.outputs}")
        our_outputs = {}
        for o in self.outputs:
            our_outputs[(o.node, o.output)] = o
        for o in other.outputs:
            if (o.node, o.output) not in our_outputs:
                raise ValueError(f"Extraneous output {o}")
            # Check if tensor dtypes match
            our_tensor = our_outputs[(o.node, o.output)].tensor
            if o.tensor.dtype != our_tensor.dtype:
                breakpoint()
                raise ValueError(f"Output {o.node}.{o.output} dtype mismatch: {o.tensor.dtype} != {our_tensor.dtype}")
            if not torch.allclose(o.tensor.to_torch(), our_outputs[(o.node, o.output)].tensor.to_torch()):
                breakpoint()
                raise ValueError(f"Output {o}={o.tensor.to_torch()} does not match our output {our_outputs[(o.node, o.output)]}={our_outputs[(o.node, o.output)].tensor.to_torch()}")

class InflightWorkManager:
    """
    Manages inflight work for a pipeline. When work has been in-flight for long
    enough, we will copy it and redistribute it. If a worker submits a result
    that is NOT inflight, we discard it. This way, we can be resilient to
    errors, with a recovery time inveresely proportional to work duplication
    between workers.
    """

    GRACE_PERIOD_S: float = 20.0

    work: dict[PartitionName, dict[str, PartitionWork]]
    times: dict[PartitionName, dict[str, float]]

    def __init__(self):
        self.work = defaultdict(dict)
        self.times = defaultdict(dict)

    def next_overdue_work(self, partition: PartitionName) -> PartitionWork | None:
        """
        Returns the next overdue work for a partition.
        """
        if partition not in self.work:
            return None

        for correlation_id, work in self.work[partition].items():
            if time.time() - self.times[partition][correlation_id] > self.GRACE_PERIOD_S:
                return work
        return None
    
    def mark_sent(self, work: PartitionWork):
        """
        Marks work as sent.
        """
        self.work[work.partition][work.correlation_id] = work
        self.times[work.partition][work.correlation_id] = time.time() # Update the time

    def acknowledge_work(self, work: PartitionWork) -> bool:
        """
        Acknowledges work for a partition. Returns True if the work was accepted (was alive). False otherwise.
        """
        # Partition doesn't exist
        if work.partition not in self.work:
            return False

        # Work doesn't exist
        if work.correlation_id not in self.work[work.partition]:
            return False

        # Grace period expired
        if time.time() - self.times[work.partition][work.correlation_id] > self.GRACE_PERIOD_S:
            return False

        # Work is alive
        del self.work[work.partition][work.correlation_id]
        del self.times[work.partition][work.correlation_id]
        return True

class ComputePipeline:
    """
    A compute pipeline stores the execution state of a compute graph.
    """

    graph: ComputeGraph
    """
    The compute graph that this pipeline is executing
    """

    partition_queues: dict[PartitionName, CorrelatedQueue]
    """
    Queues for each partition in the graph.
    """

    edge_queues: dict[ComputeGraphEdge, CorrelatedQueue]
    """
    Queues for each edge in the graph.
    """

    lock: threading.Lock

    def __init__(self, graph: ComputeGraph):
        self.graph = graph
        self.edge_queues = {}
        self.partition_queues = {}
        self.inflight_work_manager = InflightWorkManager()
        self.lock = threading.Lock()
        self._build_queues()

    def _build_queues(self):
        """
        Builds queues between cut partitions.
        """
        for partition in self.graph.get_partitions():
            edges = self.graph.identify_forward_cuts(partition)
            if len(edges) > 0:
                queue = CorrelatedQueue(list(edges), self.lock)
                self.partition_queues[partition] = queue
                for edge in edges:
                    self.edge_queues[edge] = queue

    def enqueue_input(self, input: PipelineInput):
        """
        Enqueues elements for the pipeline
        """
        # Build correlated elements
        elements = {}
        for node, tensor in input.inputs.items():
            elements[node] = CorrelatedTensor(correlation_id=input.correlation_id, tensor=tensor)

        # Ensure that all elements in the PARTITION_INPUT partition are covered
        for node in self.graph.list_partition(PARTITION_INPUT):
            if node not in elements:
                raise ValueError(f"Missing element for node {node}")

        # Ensure that all elements have the same correlation ID
        # Get the correlation ID from the first element
        if not elements:
            return
        correlation_id = next(iter(elements.values())).correlation_id

        # Check that all elements have the same correlation ID
        for node in elements:
            if elements[node].correlation_id != correlation_id:
                raise ValueError(f"All elements must have the same correlation ID")
            
        # Transform {node: element} mapping into {edge: element} mapping by finding the single edge coming from each input node
        edge_elements = {}
        for node in elements:
            forward_edges = self.graph.get_forward_edges(node)
            for edge in forward_edges:
                
                edge_elements[edge] = elements[node]

        # Enqueue elements
        for edge, element in edge_elements.items():
            self.edge_queues[edge].put(edge, element)

    def get_partition_work(self, partition: PartitionName) -> PartitionWork | None:
        """
        Gets the next partition work for a partition, or None if no work is available.
        """

        with self.lock:
            next_overdue_work = self.inflight_work_manager.next_overdue_work(partition)
            if next_overdue_work is not None:
                self.inflight_work_manager.mark_sent(next_overdue_work)
                return next_overdue_work

        elements = self.partition_queues[partition].pop(blocking=False)
        if elements is None:
            return None

        input_assignments: list[InputAssignment] = []
        correlation_id: str | None = None
        for edge, element in elements.items():
            input_assignments.append(InputAssignment(node=edge.dst, input=edge.dst_input, tensor=element.tensor))
            if correlation_id is None:
                correlation_id = element.correlation_id
            elif element.correlation_id != correlation_id:
                raise ValueError(f"All elements must have the same correlation ID")

        if correlation_id is None:
            raise ValueError("No elements found")
        
        w = PartitionWork(
            correlation_id=correlation_id,
            partition=partition,
            graph=self.graph.extract_partition(partition, include_cut_edges=False),
            inputs=input_assignments
        )

        with self.lock:
            self.inflight_work_manager.mark_sent(w)

        return w    

    def submit_partition_work(self, work: PartitionWorkResult):
        """
        Submits partition work to the pipeline.
        """

        with self.lock:
            # If the work was not alive, discard it
            if not self.inflight_work_manager.acknowledge_work(work):
                return

        for output in work.outputs:
            forward_edges = self.graph.get_forward_edges(output.node, src_output=output.output)
            for edge in forward_edges:
                self.edge_queues[edge].put(edge, CorrelatedTensor(correlation_id=work.correlation_id, tensor=output.tensor))

    def dequeue_output(self, blocking: bool = True, timeout: float | None = None) -> PipelineOutput | None:
        """
        Dequeues an output from the pipeline
        """
        # Get the next correlated set of PipelineElements
        correlated_elements = self.partition_queues[PARTITION_OUTPUT].pop(blocking=blocking, timeout=timeout)
        if correlated_elements is None:
            return None

        # Build the output
        outputs: dict[NodeName, Tensor] = {}
        for edge, element in correlated_elements.items():
            outputs[edge.dst] = element.tensor

        # Return the output
        return PipelineOutput(correlation_id=element.correlation_id, outputs=outputs)


def read_encoded_pipeline_input(offset: int, data: bytes) -> tuple[PipelineInput, int]:
    correlation_id, offset = read_encoded_string(offset, data)
    inputs_length, offset = read_be_int(offset, data)
    inputs = {}
    for _ in range(inputs_length):
        node, offset = read_encoded_string(offset, data)
        tensor, offset = read_encoded_tensor(offset, data)
        inputs[node] = tensor
    return PipelineInput(correlation_id=correlation_id, inputs=inputs), offset

def write_encoded_pipeline_input(data: bytearray, offset: int, pipeline_input: PipelineInput) -> int:
    offset = write_encoded_string(data, offset, pipeline_input.correlation_id)
    offset = write_be_int(data, offset, len(pipeline_input.inputs))
    for node, tensor in pipeline_input.inputs.items():
        offset = write_encoded_string(data, offset, node)
        offset = write_encoded_tensor(data, offset, tensor)
    return offset

def size_encoded_pipeline_input(pipeline_input: PipelineInput) -> int:
    size = size_encoded_string(pipeline_input.correlation_id)
    size += 4 # for the inputs length
    for node, tensor in pipeline_input.inputs.items():
        size += size_encoded_string(node)
        size += size_encoded_tensor(tensor)
    return size

def read_encoded_input_assignment(offset: int, data: bytes) -> tuple[InputAssignment, int]:
    node, offset = read_encoded_string(offset, data)
    input, offset = read_encoded_string(offset, data)
    tensor, offset = read_encoded_tensor(offset, data)
    return InputAssignment(node=node, input=input, tensor=tensor), offset

def write_encoded_input_assignment(data: bytearray, offset: int, input_assignment: InputAssignment) -> int:
    offset = write_encoded_string(data, offset, input_assignment.node)
    offset = write_encoded_string(data, offset, input_assignment.input)
    offset = write_encoded_tensor(data, offset, input_assignment.tensor)
    return offset

def size_encoded_input_assignment(input_assignment: InputAssignment) -> int:
    return size_encoded_string(input_assignment.node) + size_encoded_string(input_assignment.input) + size_encoded_tensor(input_assignment.tensor)

def read_encoded_output_assignment(offset: int, data: bytes) -> tuple[OutputAssignment, int]:
    node, offset = read_encoded_string(offset, data)
    output, offset = read_encoded_string(offset, data)
    tensor, offset = read_encoded_tensor(offset, data)
    return OutputAssignment(node=node, output=output, tensor=tensor), offset

def write_encoded_output_assignment(data: bytearray, offset: int, output_assignment: OutputAssignment) -> int:
    offset = write_encoded_string(data, offset, output_assignment.node)
    offset = write_encoded_string(data, offset, output_assignment.output)
    offset = write_encoded_tensor(data, offset, output_assignment.tensor)
    return offset

def size_encoded_output_assignment(output_assignment: OutputAssignment) -> int:
    return size_encoded_string(output_assignment.node) + size_encoded_string(output_assignment.output) + size_encoded_tensor(output_assignment.tensor)

def write_encoded_partition_work(data: bytearray, offset: int, partition_work: PartitionWork) -> int:
    offset = write_encoded_string(data, offset, partition_work.correlation_id)
    offset = write_encoded_string(data, offset, partition_work.partition)
    offset = partition_work.graph.encode_binary(offset, data)
    offset = write_encoded_bool(data, offset, partition_work.should_trace)
    offset = write_be_int(data, offset, len(partition_work.inputs))
    for input in partition_work.inputs:
        offset = write_encoded_input_assignment(data, offset, input)
    return offset

def size_encoded_partition_work(partition_work: PartitionWork) -> int:
    size = size_encoded_string(partition_work.correlation_id)
    size += size_encoded_string(partition_work.partition)
    size += partition_work.graph.size_binary()
    size += 1
    size += 4 # for the inputs length
    for input in partition_work.inputs:
        size += size_encoded_input_assignment(input)
    return size

def read_encoded_partition_work_result(offset: int, data: bytes) -> tuple[PartitionWorkResult, int]:
    correlation_id, offset = read_encoded_string(offset, data)
    partition, offset = read_encoded_string(offset, data)
    outputs_length, offset = read_be_int(offset, data)
    outputs = []
    for _ in range(outputs_length):
        output, offset = read_encoded_output_assignment(offset, data)
        outputs.append(output)
    return PartitionWorkResult(correlation_id=correlation_id, partition=partition, outputs=outputs), offset

def write_encoded_partition_work_result(data: bytearray, offset: int, partition_work_result: PartitionWorkResult) -> int:
    offset = write_encoded_string(data, offset, partition_work_result.correlation_id)
    offset = write_encoded_string(data, offset, partition_work_result.partition)
    offset = write_be_int(data, offset, len(partition_work_result.outputs))
    for output in partition_work_result.outputs:
        offset = write_encoded_output_assignment(data, offset, output)
    return offset

def size_encoded_partition_work_result(partition_work_result: PartitionWorkResult) -> int:
    size = size_encoded_string(partition_work_result.correlation_id)
    size += size_encoded_string(partition_work_result.partition)
    size += 4 # for the outputs length
    for output in partition_work_result.outputs:
        size += size_encoded_output_assignment(output)
    return size

def size_encoded_bool(b: bool):
    return 1
