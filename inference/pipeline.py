from __future__ import annotations

from queue import Queue, Empty
import threading
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field
from collections import defaultdict, deque

from .graph import ComputeGraph, ComputeGraphEdge, GraphEncoding, NodeName, PartitionName, PARTITION_INPUT, PARTITION_OUTPUT
from .tensor import Tensor
from .queue import CorrelatedQueue, CorrelatedTensor

import torch

class PipelineInput(BaseModel):
    """
    Input to the pipeline
    """
    correlation_id: str
    inputs: dict[NodeName, Tensor]

class PipelineOutput(BaseModel):
    """
    Output from the pipeline
    """
    correlation_id: str
    outputs: dict[NodeName, Tensor]


class InputAssignment(BaseModel):
    node: NodeName
    input: str
    tensor: Tensor

class OutputAssignment(BaseModel):
    node: NodeName
    output: str
    tensor: Tensor

class PartitionWork(BaseModel):
    """
    Work to be done for a partition
    """
    correlation_id: str
    partition: PartitionName
    graph: GraphEncoding
    inputs: list[InputAssignment]

class PartitionWorkResult(BaseModel):
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
            if not torch.allclose(o.tensor.to_torch(), our_outputs[(o.node, o.output)].tensor.to_torch()):
                raise ValueError(f"Output {o} does not match our output {our_outputs[(o.node, o.output)]}")
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

    def __init__(self, graph: ComputeGraph):
        self.graph = graph
        self.edge_queues = {}
        self.partition_queues = {}
        self._build_queues()

    def _build_queues(self):
        """
        Builds queues between cut partitions.
        """
        for partition in self.graph.get_partitions():
            edges = self.graph.identify_forward_cuts(partition)
            if len(edges) > 0:
                queue = CorrelatedQueue(list(edges))
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
        # TODO: gotta remove the peek when not debugging lmfao
        # elements = self.partition_queues[partition].peek()
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
        
        return PartitionWork(
            correlation_id=correlation_id,
            partition=partition,
            graph=self.graph.extract_partition(partition, include_cut_edges=False).encode(),
            inputs=input_assignments
        )

    def submit_partition_work(self, work: PartitionWorkResult):
        """
        Submits partition work to the pipeline.
        """
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
