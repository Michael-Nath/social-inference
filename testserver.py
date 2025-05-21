import base64
from dataclasses import dataclass
import io
import json
from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import torch
import numpy as np

from inference import (
    ModelCache, Registration, ComputePipeline, WorkerManager, 
    PartitionWork, PartitionWorkResult, PartitionName, 
    PipelineInput, PipelineOutput, Tensor, ComputeGraphBuilder,
    simulator
)

from inference import size_encoded_partition_work, size_encoded_tensor, write_encoded_partition_work, write_encoded_tensor, read_encoded_partition_work_result
from inference.graph import PARTITION_INPUT, PARTITION_OUTPUT, ComputeGraph
from inference.pipeline import (
    ComputePipeline,
    PipelineInput,
    PipelineOutput,
    PartitionWork,
    PartitionWorkResult,
)
from inference.cache import ModelCache
from inference.simulator import simulate
import tests

model_cache = ModelCache()
pipeline, g = tests.test_llama_attn()

def make_input():
    return PipelineInput(correlation_id="test_0", inputs={"x": Tensor.from_torch(torch.rand(1, 1, 2048, dtype=torch.float32))})

app = FastAPI()

# For each partition we want to make an attempt. When the client result
# mismatches, fallback to the simualtor result to prevent cascading the error.
# After all partitions have been attempted, split the ones that failed and run
# again. Keep doing this until a "point" failure is found: either a partition
# with a single node fails or a failing partition splits into two succeeding
# partitions. Report these point failures on the console.

@dataclass
class PartitionTestResult:
    partition: PartitionName
    work: PartitionWork
    result: PartitionWorkResult
    matches: bool

@dataclass
class TestContext:
    pipeline: ComputePipeline
    g: ComputeGraph
    model_cache: ModelCache
    partition_results: dict[PartitionName, PartitionTestResult]
    last_work: PartitionWork | None

test_context = TestContext(pipeline, g, model_cache, {}, None)

app.add_middleware(GZipMiddleware, minimum_size=1000)  # Compress responses larger than 1KB

@app.get("/safetensor/{model_name}/{tensor_name}")
async def get_safetensor(model_name: str, tensor_name: str):
    """
    Called by clients to get a constant tensor
    """
    # URL-decode the model_name and tensor_name as they may be URL-encoded
    model_name = base64.b64decode(model_name).decode('utf-8')
    tensor_cache = model_cache.get_cache(model_name)

    # Pretty print cache statistics
    stats = tensor_cache.get_stats()
    print(f"Cache Statistics for {model_name}:")
    print(f"  Hits: {stats.hits} ({stats.hits_bytes / (1024 * 1024):.2f} MB)")
    print(f"  Misses: {stats.misses} ({stats.misses_bytes / (1024 * 1024):.2f} MB)")
    print(f"  Evictions: {stats.evictions} ({stats.evictions_bytes / (1024 * 1024):.2f} MB)")
    print(f"  Present: {stats.present} ({stats.present_bytes / (1024 * 1024):.2f} MB)")

    with tensor_cache.get_tensor(tensor_name) as tensor:
        # tensor is torch.Tensor
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        elif tensor.dtype == torch.float16:
            tensor = tensor.to(torch.float32)
        tensor = Tensor.from_torch(tensor)
        bytes = bytearray(size_encoded_tensor(tensor))
        write_encoded_tensor(bytes, 0, tensor)
        return StreamingResponse(io.BytesIO(bytes), media_type="application/octet-stream")

@app.get("/work/")
async def get_work():
    """
    Called by clients to request inference inputs
    """
    # Find the first partition that has available work
    for partition in test_context.g.get_partitions():
        if partition == PARTITION_INPUT or partition == PARTITION_OUTPUT:
            continue
        w = test_context.pipeline.get_partition_work(partition)
        if w is not None:
            test_context.last_work = w
            bytes = bytearray(size_encoded_partition_work(w))
            write_encoded_partition_work(bytes, 0, w)
            return StreamingResponse(io.BytesIO(bytes), media_type="application/octet-stream")
    return StreamingResponse(io.BytesIO(b""), status_code=404, media_type="application/octet-stream")

@app.post("/work", response_model=bool)
async def submit_work(req: Request):
    """
    Called by clients to submit inference results
    """

    # Buffer request
    body = b""
    async for chunk in req.stream():
        body += chunk

    # Parse JSON
    work, _ = read_encoded_partition_work_result(0, body)

    correct_result = work
    failure = False
    try:
        gt = simulator.simulate(test_context.last_work, test_context.model_cache)
        gt.close_to(work)
    except ValueError as e:
        correct_result = gt
        failure = True

    test_context.partition_results[work.partition] = PartitionTestResult(
        partition=work.partition,
        work=test_context.last_work,
        result=work,
        matches=not failure
    )

    test_context.pipeline.submit_partition_work(correct_result)

    # Check if we have finished
    if test_context.pipeline.dequeue_output() is not None:

        # Split failing partitions
        mutated = False
        for partition in test_context.g.get_partitions():
            if partition == PARTITION_INPUT or partition == PARTITION_OUTPUT:
                continue

            if not test_context.partition_results[partition].matches:
                if len(test_context.g.list_partition(partition)) > 1:
                    print(f"Splitting partition {partition}")
                    test_context.g.split_partition(partition)
                    mutated = True

        print("=" * 5, "results so far", "=" * 5)
        for partition, result in test_context.partition_results.items():
            if not result.matches:
                print(f"Partition {partition} failed")
        print("=" * 20)
        for partition in test_context.g.get_partitions():
            print(f"Partition {partition}: {len(test_context.g.list_partition(partition))}")
        print("=" * 20)

        # Rebuild pipeline
        test_context.pipeline = ComputePipeline(test_context.g)

        # Requeue an input 
        test_context.pipeline.enqueue_input(make_input())

        return mutated
    return True

# Mount the frontend directory to serve static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, log_level="debug", host="0.0.0.0", port=8080)