import threading

from pydantic import BaseModel
from .graph import PartitionName, ComputeGraph, PARTITION_INPUT, PARTITION_OUTPUT

class RegistrationRequest(BaseModel):
    is_mobile: bool

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

    def register(self, req: RegistrationRequest) -> Registration:
        """
        Requests registration for a worker.
        """
        with self.lock:
            # Find partition with minimum workers, skipping pre/post for mobile
            eligible_partitions = self.assignmentCounts
            if req.is_mobile:
                eligible_partitions = {k: v for k, v in self.assignmentCounts.items() 
                                    if 'pre' not in k and 'post' not in k}

            min_partition = min(eligible_partitions.items(), key=lambda x: x[1])
            partition_name = min_partition[0]
            
            # Increment worker count for this partition
            self.assignmentCounts[partition_name] += 1
            
            for k,v in self.assignmentCounts.items():
                print(f"Partition {k} has {v} workers")
                
            return Registration(partition=partition_name)