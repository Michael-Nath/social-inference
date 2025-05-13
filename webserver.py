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

CHECK_WORK = True

# model_cache = ModelCache()
model_cache = None

def simple_two_node():
    g = ComputeGraphBuilder()
    x = g.input("x")
    y = g.input("y")
    with g.partition("p0"):
        matmul = g.matmul("matmul", x, y)
        add = g.add("add1", matmul, x)
    z = g.output("output", add)
    return g.build()

def test_softmax():
    g = ComputeGraphBuilder()
    x = g.input("x")
    with g.partition("p0"):
        two = g.fixed("two", torch.tensor([2]))
        softmax = g.softmax("softmax", x, two)
    y = g.output("output", softmax)
    g = g.build()

    pipeline = ComputePipeline(g)
    for i in range(20):
        pipeline.enqueue_input(PipelineInput(
            correlation_id=f"{i}",
            inputs={
                "x": Tensor.from_torch(torch.randn(4, 4, 4, 4)),
            },
        ))
    return pipeline, g

def test_unsqueeze():
    g = ComputeGraphBuilder()
    x = g.input("x")
    with g.partition("p0"):
        zero = g.fixed("zero", torch.tensor([0]))
        unsqueeze = g.unsqueeze("unsqueeze", x, zero)
    y = g.output("output", unsqueeze)
    g = g.build()

    pipeline = ComputePipeline(g)
    for i in range(20):
        pipeline.enqueue_input(PipelineInput(
            correlation_id=f"{i}",
            inputs={
                "x": Tensor.from_torch(torch.randn(4, 4, 4, 4)),
            },
        ))
    return pipeline, g

def test_broadcast():
    g = ComputeGraphBuilder()
    x = g.input("x")
    with g.partition("p0"):
        two = g.fixed("two", torch.tensor([2], dtype=torch.int32))
        ten = g.fixed("ten", torch.tensor([10], dtype=torch.int32))
        unsqueezed = g.unsqueeze("unsqueeze", x, two)
        broadcast = g.broadcast("broadcast", unsqueezed, two, ten)
    y = g.output("output", broadcast)
    g = g.build()

    pipeline = ComputePipeline(g)
    for i in range(20):
        pipeline.enqueue_input(PipelineInput(
            correlation_id=f"{i}",
            inputs={
                "x": Tensor.from_torch(torch.randn(4, 4, 4, 4)),
            },
        ))
    return pipeline, g

pipeline, g = test_broadcast()
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