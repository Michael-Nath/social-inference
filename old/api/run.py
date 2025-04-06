from dataclasses import dataclass
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import numpy as np
from typing import List, Dict

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add GZip compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)  # Compress responses larger than 1KB

@dataclass
class Dimensions:
    vocab: int
    embedding: int

def dims_from_model(path: str):
    # path is an identifier of some model
    return Dimensions(
        vocab = 10000,
        embedding = 1024
    )

dims = dims_from_model("")

TENSORS = [
    np.random.randn(dims.vocab, dims.embedding),  # embedding matrix
    np.random.randn(dims.embedding, 4 * dims.embedding),  # mlp-1
    np.random.randn(4 * dims.embedding, 4 * dims.embedding),  # mlp-2
    np.random.randn(4 * dims.embedding, dims.embedding),  # mlp-3
]

# Track which client has which tensor
client_assignments: Dict[str, int] = {}
next_tensor_idx = 0

class TensorResponse(BaseModel):
    tensor: List[float]
    shape: List[int]
    client_id: str

@app.get("/get_tensor", response_model=TensorResponse)
async def get_tensor(request: Request):
    global next_tensor_idx
    
    # Use IP address as client identifier
    client_id = request.client.host
    
    # Assign tensor if this is a new client
    if client_id not in client_assignments:
        client_assignments[client_id] = next_tensor_idx
        next_tensor_idx = (next_tensor_idx + 1) % len(TENSORS)
    
    # Get the assigned tensor
    tensor_idx = client_assignments[client_id]
    tensor = TENSORS[tensor_idx]
    
    return {
        "tensor": tensor.flatten().tolist(),
        "shape": list(tensor.shape),
        "client_id": client_id
    }

@app.get("/stats")
async def get_stats():
    return {
        "assignments": client_assignments,
        "tensor_shapes": [tensor.shape for tensor in TENSORS],
        "tensor_sizes_mb": [tensor.nbytes / (1024 * 1024) for tensor in TENSORS]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)