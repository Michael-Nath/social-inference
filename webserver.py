from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
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

# model_cache = ModelCache()
model_cache = None

pipeline, g = tests.test_llama_model()
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
    w = pipeline.get_partition_work(partition_name)
    print("Done packing work")
    if CHECK_WORK:
        inflight_work[(partition_name, w.correlation_id)] = w
    print("Done inflight work")
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