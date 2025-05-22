import threading

from pydantic import BaseModel
from .graph import PartitionName, ComputeGraph, PARTITION_INPUT, PARTITION_OUTPUT

class Registration(BaseModel):
    partition: PartitionName

class WorkerManager:
    graph: ComputeGraph
    assignmentCounts: dict[PartitionName, int]
    lock: threading.Lock

    def __init__(self, graph: ComputeGraph):
        self.graph = graph
        self.assignmentCounts = {}
        for p in (graph.get_partitions() - {PARTITION_INPUT, PARTITION_OUTPUT}):
            self.assignmentCounts[p] = 0

        

        if not self.assignmentCounts:
            raise ValueError("No partitions to assign")
        self.lock = threading.Lock()

    def register(self) -> Registration:
        """
        Requests registration for a worker.

        For now, just assigns to the partition with the least workers.
        """
        with self.lock:
            # Find partition with minimum workers
            min_partition = min(self.assignmentCounts.items(), key=lambda x: x[1])
            partition_name = min_partition[0]
            # Increment worker count for this partition
            self.assignmentCounts[partition_name] += 1
            for k,v in self.assignmentCounts.items():
                print(f"Partition {k} has {v} workers")
            return Registration(partition=partition_name)