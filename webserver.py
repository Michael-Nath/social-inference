from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch

from inference import (
    ModelCache, Registration, ComputePipeline, WorkerManager, 
    PartitionWork, PartitionWorkResult, PartitionName, 
    PipelineInput, PipelineOutput, Tensor, ComputeGraphBuilder
)
from models import simple_matmul

model_cache = ModelCache()

def simple_two_node():
    g = ComputeGraphBuilder()
    x = g.input("x")
    y = g.input("y")
    with g.partition("p0"):
        matmul = g.matmul("matmul", x, y)
        add = g.add("add", matmul, x)
    z = g.output("output", add)
    return g.build()

g = simple_two_node()
pipeline = ComputePipeline(g)
pipeline.enqueue_input(PipelineInput(
    correlation_id="1",
    inputs={
        "x": Tensor.from_torch(torch.randn(16,16)),
        "y": Tensor.from_torch(torch.randn(16,16)),
    },
))
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

@app.get("/constant/{model_name}/{tensor_name}", response_model=Tensor)
async def get_constant(model_name: str, tensor_name: str):
    """
    Called by clients to get a constant tensor
    """
    tensor_cache = model_cache.get_cache(model_name)
    with tensor_cache.get_tensor(tensor_name) as tensor:
        return Tensor.from_torch(tensor)

@app.get("/work/{partition_name}", response_model=PartitionWork | None)
async def get_work(partition_name: PartitionName):
    """
    Called by clients to request inference inputs
    """
    return pipeline.get_partition_work(partition_name)

@app.post("/work", response_model=PartitionWorkResult)
async def submit_work(work: PartitionWorkResult):
    """
    Called by clients to submit inference results
    """
    return pipeline.submit_partition_work(work)

# Mount the frontend directory to serve static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)