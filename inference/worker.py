import asyncio
import threading
import time

from pydantic import BaseModel
from .graph import PartitionName, ComputeGraph, PARTITION_INPUT, PARTITION_OUTPUT

class RegistrationRequest(BaseModel):
    is_mobile: bool

class Registration(BaseModel):
    partition: PartitionName
    session_id: str

class WorkerManager:
    graph: ComputeGraph
    assignmentCounts: dict[PartitionName, int]
    lock: threading.Lock

    def __init__(self, graph: ComputeGraph):
        self.graph = graph
        self.session_id = str(time.time())
        self.assignmentCounts = {}
        for p in (graph.get_partitions() - {PARTITION_INPUT, PARTITION_OUTPUT}):
            self.assignmentCounts[p] = 0

        if not self.assignmentCounts:
            raise ValueError("No partitions to assign")
        self.lock = threading.Lock()

    def revived(self, partition_name: PartitionName, cand_session_id: str):
        """
        Called when a worker observes a prior partition
        """
        if (cand_session_id != self.session_id):
            # client is not on same session as the server, cannot decrement anything
            # because the client's partition in browser storage is not accurate
            return
        # the client is on the same session as the server and the client has a valid assignment
        with self.lock:
            if partition_name in self.assignmentCounts:
                # client "gives up" this partition; the coordinator can either give this back in
                # subsequent register(), or assign another partition
                self.assignmentCounts[partition_name] -= 1

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
                # For mobile, prefer smaller partitions
                partition_keys = list(eligible_partitions.keys())
                partition_keys.sort(key=lambda p: (self.assignmentCounts[p], len(self.graph.list_partition(p))))
            else:
                # For desktop, prefer larger partitions
                partition_keys = list(eligible_partitions.keys())
                partition_keys.sort(key=lambda p: (self.assignmentCounts[p], -len(self.graph.list_partition(p))))

            partition_name = partition_keys[0]
            
            # Increment worker count for this partition
            self.assignmentCounts[partition_name] += 1
            
            for k,v in self.assignmentCounts.items():
                print(f"Partition {k} has {v} workers")
                
            return Registration(partition=partition_name, session_id=self.session_id)
        

class AsyncWorkerManager:
    graph: ComputeGraph
    assignmentCounts: dict[PartitionName, int]
    lock: asyncio.Lock

    def __init__(self, graph: ComputeGraph):
        self.graph = graph
        self.assignmentCounts = {}
        for p in (graph.get_partitions() - {PARTITION_INPUT, PARTITION_OUTPUT}):
            self.assignmentCounts[p] = 0

        if not self.assignmentCounts:
            raise ValueError("No partitions to assign")
        self.lock = asyncio.Lock()

    async def revived(self, partition_name: PartitionName):
        """
        Called when a worker observes a prior partition
        """
        async with self.lock:
            if partition_name in self.assignmentCounts:
                self.assignmentCounts[partition_name] -= 1

    async def register(self, req: RegistrationRequest) -> Registration:
        """
        Requests registration for a worker.
        """
        async with self.lock:
            # Find partition with minimum workers, skipping pre/post for mobile
            eligible_partitions = self.assignmentCounts
            if req.is_mobile:
                eligible_partitions = {k: v for k, v in self.assignmentCounts.items() 
                                    if 'pre' not in k and 'post' not in k}
                # For mobile, prefer smaller partitions
                partition_keys = list(eligible_partitions.keys())
                partition_keys.sort(key=lambda p: (self.assignmentCounts[p], len(self.graph.list_partition(p))))
            else:
                # For desktop, prefer larger partitions
                partition_keys = list(eligible_partitions.keys())
                partition_keys.sort(key=lambda p: (self.assignmentCounts[p], -len(self.graph.list_partition(p))))

            partition_name = partition_keys[0]
            
            # Increment worker count for this partition
            self.assignmentCounts[partition_name] += 1
            
            for k,v in self.assignmentCounts.items():
                print(f"Partition {k} has {v} workers")
                
            return Registration(partition=partition_name)
            
