import base64
import io
import json
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch

from inference import (
    ModelCache, Registration, ComputePipeline, WorkerManager, 
    PartitionWork, PartitionWorkResult, PartitionName, 
    PipelineInput, PipelineOutput, Tensor, ComputeGraphBuilder,
    simulator
)

import tests

# Import test functions from tests.py
import tests

CHECK_WORK = True

model_cache = ModelCache()
pipeline, g = tests.test_safetensor()
worker_manager = WorkerManager(g)

app = FastAPI()

# Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

inflight_work = {}

app.add_middleware(GZipMiddleware, minimum_size=1000)  # Compress responses larger than 1KB

@app.post("/register", response_model=Registration)
async def register():
    """
    Called by clients to register their capabilities and request assignment to work.
    """
    return worker_manager.register()

@app.post("/input")
async def push_input(input: PipelineInput):
    """
    Called by clients to push inference inputs
    """
    pipeline.enqueue_input(input)

@app.get("/output", response_model=PipelineOutput | None)
async def get_output():
    """
    Called by clients to get inference outputs
    """
    return pipeline.dequeue_output(blocking=False)

class SafetensorHeader(BaseModel):
    dtype: str
    shape: list[int]

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
        bytes = tensor.detach().cpu().numpy().tobytes()
        dtype_str = str(tensor.dtype)
        if dtype_str.startswith('torch.'):
            dtype_str = dtype_str[6:]  # Remove 'torch.' prefix
        header = SafetensorHeader(
            dtype=dtype_str,
            shape=list(tensor.shape),
        )
        # Convert to JSON then prefix byte array
        header_bytes = header.model_dump_json().encode('utf-8')
        header_size = len(header_bytes).to_bytes(4, byteorder='big')
        return StreamingResponse(io.BytesIO(header_size + header_bytes + bytes), media_type="application/octet-stream")

@app.get("/work/{partition_name}", response_model=PartitionWork | None)
async def get_work(partition_name: PartitionName):
    """
    Called by clients to request inference inputs
    """
    w = pipeline.get_partition_work(partition_name)
    if CHECK_WORK:
        inflight_work[(partition_name, w.correlation_id)] = w
    return w

@app.post("/work")
async def submit_work(work: PartitionWorkResult):
    """
    Called by clients to submit inference results
    """
    # Check work
    if CHECK_WORK:
        gt = simulator.simulate(inflight_work[(work.partition, work.correlation_id)], model_cache)
        gt.close_to(work)
    return pipeline.submit_partition_work(work)



# Mount the frontend directory to serve static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)