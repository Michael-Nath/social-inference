from typing import Literal, Annotated, Union
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field

import numpy as np
import torch
from models import ModelCache

model_cache = ModelCache()
app = FastAPI()

# Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

app.add_middleware(GZipMiddleware, minimum_size=1000)  # Compress responses larger than 1KB



class ClientCapabilities(BaseModel):
    maxBufferSize: int
    """
    Max buffer size in bytes
    """

class RegisterRequest(BaseModel):
    capabilities: ClientCapabilities

class Tensor(BaseModel):
    elements: list[float]
    shape: list[int]

    @classmethod
    def from_numpy(cls, np_array: np.ndarray):
        return cls(elements=np_array.flatten().tolist(), shape=list(np_array.shape))
    
    @classmethod
    def from_torch(cls, torch_tensor):
        np_array = torch_tensor.detach().cpu().numpy()
        return cls(elements=np_array.flatten().tolist(), shape=list(np_array.shape))

class MatMulOperationAssignment(BaseModel):
    operation: Literal["matmul"]
    matrix: Tensor

OperationData = Annotated[Union[MatMulOperationAssignment], Field(discriminator="operation")]

class RegisterResponse(BaseModel):
    operation: OperationData
    """
    Operation assigned to client
    """

@app.post("/register", response_model=RegisterResponse)
async def register(request: RegisterRequest):
    """
    Called by clients to register their capabilities and request assignment to work.

    {
    operation: {
    operation: "matmul",
    matrix: {
    elements: [....]
    shape: [32,32]
    }
    }
    }
    """
    cache = model_cache.get_cache("meta-llama/Llama-3.2-1B")
    with cache.get_tensor("model.layers.0.self_attn.k_proj.weight") as t:
        t = t.float()[:64,:64]
        print(t.shape)
        print(t.dtype)
        return RegisterResponse(
            operation=MatMulOperationAssignment(
                operation="matmul",
                matrix=Tensor.from_torch(t)
            )
        )
class WorkResponse(BaseModel):
    pass
@app.get("/work", response_model=WorkResponse)
async def get_work():
    """
    Called by clients to request inference inputs
    """
    pass

# Mount the frontend directory to serve static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)