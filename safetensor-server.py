import base64
import io
import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch

from inference import (
    ModelCache, AsyncModelCache, Registration, ComputePipeline, WorkerManager, NextTokenManager,
    PartitionName, SingleStepChunk, Prompt,
    PipelineOutput, Tensor, size_encoded_partition_work, write_encoded_partition_work, size_encoded_tensor, write_encoded_tensor,
    read_encoded_partition_work_result, simulator
)

from inference.graph import PARTITION_INPUT, PARTITION_OUTPUT
from inference.builds import build_llaam_causal_mp

from inference.pipeline import CorrelationResponse
from inference.worker import AsyncWorkerManager, RegistrationRequest
import tests

model_cache = AsyncModelCache()
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def stream_bytes(bytes: bytes, chunk_size: int = 1024 * 500):
    n_bytes = len(bytes)
    for i in range(0, n_bytes, chunk_size):
        start = i
        end = min(i + chunk_size, n_bytes)
        yield bytes[start:end]

@app.get("/{model_name}/{tensor_name}")
async def get_safetensor(model_name: str, tensor_name: str):
    """
    Called by clients to get a constant tensor
    """
    # URL-decode the model_name and tensor_name as they may be URL-encoded
    model_name = base64.b64decode(model_name).decode('utf-8')
    tensor_cache = model_cache.get_cache(model_name)

    # Pretty print cache statistics
    stats = await tensor_cache.get_stats()
    print(f"Cache Statistics for {model_name}:")
    print(f"  Hits: {stats.hits} ({stats.hits_bytes / (1024 * 1024):.2f} MB)")
    print(f"  Misses: {stats.misses} ({stats.misses_bytes / (1024 * 1024):.2f} MB)")
    print(f"  Evictions: {stats.evictions} ({stats.evictions_bytes / (1024 * 1024):.2f} MB)")
    print(f"  Present: {stats.present} ({stats.present_bytes / (1024 * 1024):.2f} MB)")

    async with tensor_cache.get_tensor(tensor_name) as tensor:
        # tensor is torch.Tensor
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        elif tensor.dtype == torch.float16:
            tensor = tensor.to(torch.float32)
        tensor = Tensor.from_torch(tensor)
    tensorBytes = bytearray(size_encoded_tensor(tensor))
    write_encoded_tensor(tensorBytes, 0, tensor)
    tensorBytes = bytes(tensorBytes)

    # For large files, use chunked transfer encoding
    return StreamingResponse(
        stream_bytes(tensorBytes),  # Send the entire buffer as one chunk
        media_type="application/octet-stream",
        headers={
            "Content-Length": str(len(tensorBytes)),
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
