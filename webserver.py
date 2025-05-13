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
        two = g.fixed("two", torch.tensor([2], dtype=torch.int32))
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
        zero = g.fixed("zero", torch.tensor([0], dtype=torch.int32))
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

def test_cat():
    g = ComputeGraphBuilder()
    a_in = g.input("a")
    b_in = g.input("b")
    with g.partition("p0"):
        # Concatenate along dimension 1
        dim_tensor = g.fixed("dim", torch.tensor([1], dtype=torch.int32))
        cat_node = g.cat("cat_op", a_in, b_in, dim_tensor)
    output = g.output("output", cat_node)
    graph = g.build()

    pipeline = ComputePipeline(graph)
    # Example input: two tensors of shape (2, 3, 4)
    # Expected output shape after cat along dim 1: (2, 6, 4)
    for i in range(5): # Let's create a few work items
        pipeline.enqueue_input(PipelineInput(
            correlation_id=f"cat_test_{i}",
            inputs={
                "a": Tensor.from_torch(torch.randn(2, 3, 4)),
                "b": Tensor.from_torch(torch.randn(2, 3, 4)),
            },
        ))
    return pipeline, graph

def test_math_ops():
    g = ComputeGraphBuilder()
    # Integer Test Path
    int_a_in = g.input("int_a")
    int_b_in = g.input("int_b")
    # Float Test Path
    float_a_in = g.input("float_a")
    float_b_in = g.input("float_b")

    with g.partition("p0_math_ops"):
        # Integer operations
        int_div_node = g.div("int_div", int_a_in, int_b_in)
        int_floor_node = g.floor("int_floor", int_div_node) # Floor of integer division
        int_ceil_node = g.ceil("int_ceil", int_div_node)   # Ceil of integer division

        # Float operations
        float_div_node = g.div("float_div", float_a_in, float_b_in)
        float_floor_node = g.floor("float_floor", float_div_node)
        float_ceil_node = g.ceil("float_ceil", float_div_node)

        # Mixed: floor a float, ceil a float, then div them
        # For this, let's use float_a_in directly for floor/ceil
        intermediate_float_floor = g.floor("intermediate_floor", float_a_in)
        intermediate_float_ceil = g.ceil("intermediate_ceil", float_a_in)
        mixed_div_node = g.div("mixed_div", intermediate_float_ceil, intermediate_float_floor) # e.g. ceil(7.5)/floor(7.5) = 8/7

    # Outputs
    g.output("int_floor_out", int_floor_node)
    g.output("int_ceil_out", int_ceil_node)
    g.output("float_floor_out", float_floor_node)
    g.output("float_ceil_out", float_ceil_node)
    g.output("mixed_div_out", mixed_div_node)
    
    graph = g.build()
    pipeline = ComputePipeline(graph)

    # Enqueue test inputs
    # Integers: 7 / 3 = 2.33... floor=2, ceil=3
    # Floats: 7.5 / 2.5 = 3.0. floor=3, ceil=3
    # Mixed div: float_a = 7.5. ceil(7.5)=8, floor(7.5)=7. 8/7 = 1.14...
    for i in range(10):
        pipeline.enqueue_input(PipelineInput(
            correlation_id=f"math_ops_test_{i}",
            inputs={
                "int_a": Tensor.from_torch(torch.tensor([[7, 10]], dtype=torch.int32)),
                "int_b": Tensor.from_torch(torch.tensor([[3, 4]], dtype=torch.int32)),
                "float_a": Tensor.from_torch(torch.tensor([[7.5, 10.8]], dtype=torch.float32)),
                "float_b": Tensor.from_torch(torch.tensor([[2.5, 2.0]], dtype=torch.float32)),
            },
        ))
    return pipeline, graph

pipeline, g = test_math_ops()
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