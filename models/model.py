from queue import Empty, Queue
import threading
from typing import Literal, Annotated, Union

from pydantic import BaseModel, Field
import torch

from models.cache import ModelCache
from .tensor import Tensor
from dataclasses import dataclass


class Work(BaseModel):
    order: int
    operation: str
    operands: list[Tensor]

class WorkResult(BaseModel):
    order: int
    operation: str
    results: list[Tensor]

@dataclass
class PipelineData:
    order: int
    data: list[Tensor]

class ClientCapabilities(BaseModel):
    maxBufferSize: int
    """
    Max buffer size in bytes
    """

class RegistrationRequest(BaseModel):
    capabilities: ClientCapabilities

class MatmulRegistration(BaseModel):
    operation: Union[Literal["matmul0"] | Literal["matmul1"]]
    matrix: Tensor

Registration = Annotated[Union[MatmulRegistration], Field(discriminator="operation")]

class Pipeline:
    input_queue: Queue
    step1_workers: int
    intermediate_queue: Queue
    step2_workers: int
    output_queue: Queue

    model_cache: ModelCache
    lock: threading.Lock

    def __init__(self):
        self.input_queue = Queue()
        self.step1_workers = 0
        self.intermediate_queue = Queue()
        self.step2_workers = 0
        self.output_queue = Queue()
        self.model_cache = ModelCache()
        self.lock = threading.Lock()

        for i in range(10):
            self.input_queue.put(PipelineData(order=i, data=[Tensor.from_torch(torch.randn(64,64))]))

    def get_queues(self) -> list[Queue]:
        return [self.input_queue, self.intermediate_queue, self.output_queue]

    def register(self, request: RegistrationRequest) -> Registration:
        """
        Requests a registration for a worker
        """
        with self.lock:
            if self.step1_workers <= self.step2_workers:
                matrix = "model.layers.0.self_attn.k_proj.weight"
                operation = "matmul0"
                self.step1_workers += 1
            else:
                matrix = "model.layers.1.self_attn.k_proj.weight"
                operation = "matmul1"
                self.step2_workers += 1
        cache = self.model_cache.get_cache("meta-llama/Llama-3.2-1B")
        with cache.get_tensor(matrix) as t:
            t = t.float()[:64,:64]
            return MatmulRegistration(operation=operation, matrix=Tensor.from_torch(t))

    def get_work(self, operation: str) -> Work | None:
        if operation == "matmul0":
            queue = self.input_queue
        elif operation == "matmul1":
            queue = self.intermediate_queue
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        try:
            popped = queue.get(block=False)
        except Empty:
            return None

        return Work(order=popped.order, operation=operation, operands=[popped.data])

    def submit_work(self, work: WorkResult):
        if work.operation == "matmul0":
            self.intermediate_queue.put(PipelineData(order=work.order, data=work.results))
        elif work.operation == "matmul1":
            self.output_queue.put(PipelineData(order=work.order, data=work.results))
        else:
            raise ValueError(f"Unknown operation: {work.operation}")