import base64
import io
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch

from inference import (
    ModelCache, Registration, ComputePipeline, WorkerManager, 
    PartitionWork, PartitionWorkResult, PartitionName, SingleStepChunk,
    PipelineInput, PipelineOutput, Tensor, ComputeGraphBuilder, size_encoded_partition_work, write_encoded_partition_work, size_encoded_tensor, write_encoded_tensor,
    read_encoded_partition_work_result, simulator
)

import tests


llama_model_param_keys = ['embed_tokens.weight', 'layers.0.self_attn.q_proj.weight', 'layers.0.self_attn.k_proj.weight', 'layers.0.self_attn.v_proj.weight', 'layers.0.self_attn.o_proj.weight', 'layers.0.mlp.gate_proj.weight', 'layers.0.mlp.up_proj.weight', 'layers.0.mlp.down_proj.weight', 'layers.0.input_layernorm.weight', 'layers.0.post_attention_layernorm.weight', 'layers.1.self_attn.q_proj.weight', 'layers.1.self_attn.k_proj.weight', 'layers.1.self_attn.v_proj.weight', 'layers.1.self_attn.o_proj.weight', 'layers.1.mlp.gate_proj.weight', 'layers.1.mlp.up_proj.weight', 'layers.1.mlp.down_proj.weight', 'layers.1.input_layernorm.weight', 'layers.1.post_attention_layernorm.weight', 'layers.2.self_attn.q_proj.weight', 'layers.2.self_attn.k_proj.weight', 'layers.2.self_attn.v_proj.weight', 'layers.2.self_attn.o_proj.weight', 'layers.2.mlp.gate_proj.weight', 'layers.2.mlp.up_proj.weight', 'layers.2.mlp.down_proj.weight', 'layers.2.input_layernorm.weight', 'layers.2.post_attention_layernorm.weight', 'layers.3.self_attn.q_proj.weight', 'layers.3.self_attn.k_proj.weight', 'layers.3.self_attn.v_proj.weight', 'layers.3.self_attn.o_proj.weight', 'layers.3.mlp.gate_proj.weight', 'layers.3.mlp.up_proj.weight', 'layers.3.mlp.down_proj.weight', 'layers.3.input_layernorm.weight', 'layers.3.post_attention_layernorm.weight', 'layers.4.self_attn.q_proj.weight', 'layers.4.self_attn.k_proj.weight', 'layers.4.self_attn.v_proj.weight', 'layers.4.self_attn.o_proj.weight', 'layers.4.mlp.gate_proj.weight', 'layers.4.mlp.up_proj.weight', 'layers.4.mlp.down_proj.weight', 'layers.4.input_layernorm.weight', 'layers.4.post_attention_layernorm.weight', 'layers.5.self_attn.q_proj.weight', 'layers.5.self_attn.k_proj.weight', 'layers.5.self_attn.v_proj.weight', 'layers.5.self_attn.o_proj.weight', 'layers.5.mlp.gate_proj.weight', 'layers.5.mlp.up_proj.weight', 'layers.5.mlp.down_proj.weight', 'layers.5.input_layernorm.weight', 'layers.5.post_attention_layernorm.weight', 'layers.6.self_attn.q_proj.weight', 'layers.6.self_attn.k_proj.weight', 'layers.6.self_attn.v_proj.weight', 'layers.6.self_attn.o_proj.weight', 'layers.6.mlp.gate_proj.weight', 'layers.6.mlp.up_proj.weight', 'layers.6.mlp.down_proj.weight', 'layers.6.input_layernorm.weight', 'layers.6.post_attention_layernorm.weight', 'layers.7.self_attn.q_proj.weight', 'layers.7.self_attn.k_proj.weight', 'layers.7.self_attn.v_proj.weight', 'layers.7.self_attn.o_proj.weight', 'layers.7.mlp.gate_proj.weight', 'layers.7.mlp.up_proj.weight', 'layers.7.mlp.down_proj.weight', 'layers.7.input_layernorm.weight', 'layers.7.post_attention_layernorm.weight', 'layers.8.self_attn.q_proj.weight', 'layers.8.self_attn.k_proj.weight', 'layers.8.self_attn.v_proj.weight', 'layers.8.self_attn.o_proj.weight', 'layers.8.mlp.gate_proj.weight', 'layers.8.mlp.up_proj.weight', 'layers.8.mlp.down_proj.weight', 'layers.8.input_layernorm.weight', 'layers.8.post_attention_layernorm.weight', 'layers.9.self_attn.q_proj.weight', 'layers.9.self_attn.k_proj.weight', 'layers.9.self_attn.v_proj.weight', 'layers.9.self_attn.o_proj.weight', 'layers.9.mlp.gate_proj.weight', 'layers.9.mlp.up_proj.weight', 'layers.9.mlp.down_proj.weight', 'layers.9.input_layernorm.weight', 'layers.9.post_attention_layernorm.weight', 'layers.10.self_attn.q_proj.weight', 'layers.10.self_attn.k_proj.weight', 'layers.10.self_attn.v_proj.weight', 'layers.10.self_attn.o_proj.weight', 'layers.10.mlp.gate_proj.weight', 'layers.10.mlp.up_proj.weight', 'layers.10.mlp.down_proj.weight', 'layers.10.input_layernorm.weight', 'layers.10.post_attention_layernorm.weight', 'layers.11.self_attn.q_proj.weight', 'layers.11.self_attn.k_proj.weight', 'layers.11.self_attn.v_proj.weight', 'layers.11.self_attn.o_proj.weight', 'layers.11.mlp.gate_proj.weight', 'layers.11.mlp.up_proj.weight', 'layers.11.mlp.down_proj.weight', 'layers.11.input_layernorm.weight', 'layers.11.post_attention_layernorm.weight', 'layers.12.self_attn.q_proj.weight', 'layers.12.self_attn.k_proj.weight', 'layers.12.self_attn.v_proj.weight', 'layers.12.self_attn.o_proj.weight', 'layers.12.mlp.gate_proj.weight', 'layers.12.mlp.up_proj.weight', 'layers.12.mlp.down_proj.weight', 'layers.12.input_layernorm.weight', 'layers.12.post_attention_layernorm.weight', 'layers.13.self_attn.q_proj.weight', 'layers.13.self_attn.k_proj.weight', 'layers.13.self_attn.v_proj.weight', 'layers.13.self_attn.o_proj.weight', 'layers.13.mlp.gate_proj.weight', 'layers.13.mlp.up_proj.weight', 'layers.13.mlp.down_proj.weight', 'layers.13.input_layernorm.weight', 'layers.13.post_attention_layernorm.weight', 'layers.14.self_attn.q_proj.weight', 'layers.14.self_attn.k_proj.weight', 'layers.14.self_attn.v_proj.weight', 'layers.14.self_attn.o_proj.weight', 'layers.14.mlp.gate_proj.weight', 'layers.14.mlp.up_proj.weight', 'layers.14.mlp.down_proj.weight', 'layers.14.input_layernorm.weight', 'layers.14.post_attention_layernorm.weight', 'layers.15.self_attn.q_proj.weight', 'layers.15.self_attn.k_proj.weight', 'layers.15.self_attn.v_proj.weight', 'layers.15.self_attn.o_proj.weight', 'layers.15.mlp.gate_proj.weight', 'layers.15.mlp.up_proj.weight', 'layers.15.mlp.down_proj.weight', 'layers.15.input_layernorm.weight', 'layers.15.post_attention_layernorm.weight', 'norm.weight']
llama_layer_one_param_keys = ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight', 'self_attn.o_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'input_layernorm.weight', 'post_attention_layernorm.weight']
    
# ]
model_cache = ModelCache()
# pipeline, g = tests.test_safetensor()
# pipeline, g = tests.test_llama_layernorm()
# pipeline, g = tests.test_llama_attn()
pipeline, g = tests.test_llama_model()
# pipeline, g = tests.test_llama_causal()
# pipeline, g = tests.test_transpose()
# pipeline, g = tests.test_llama_layer(llama_layer_one_param_keys, "meta-llama/Llama-3.2-1B", idx=0)
# pipeline, g = tests.test_index_select()
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
sim_results = {}

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

async def stream_bytes(bytes: bytes, chunk_size: int = 1024 * 500):
    n_bytes = len(bytes)
    for i in range(0, n_bytes, chunk_size):
        start = i
        end = min(i + chunk_size, n_bytes)
        yield bytes[start:end]

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

@app.get("/work/{partition_name}")
async def get_work(partition_name: PartitionName):
    """
    Called by clients to request inference inputs
    """
    w = pipeline.get_partition_work(partition_name)
    if w is not None:
        w.should_trace = True
        inflight_work[(w.partition, w.correlation_id)] = w
        bytes = bytearray(size_encoded_partition_work(w))
        write_encoded_partition_work(bytes, 0, w)
        return StreamingResponse(io.BytesIO(bytes), media_type="application/octet-stream")
    return StreamingResponse(io.BytesIO(b""), status_code=404, media_type="application/octet-stream")

@app.post("/work")
async def submit_work(req: Request):
    """
    Called by clients to submit inference results
    """
    body = bytearray()
    async for chunk in req.stream():
        body.extend(chunk)

    # Parse JSON
    work, _ = read_encoded_partition_work_result(0, body)
    return pipeline.submit_partition_work(work)

@app.post("/check-work")
async def check_work(req: Request):
    body = bytearray()
    async for chunk in req.stream():
        body.extend(chunk)

    work, _ = SingleStepChunk.decode(0, body)
    if (work.partition, work.correlation_id) not in sim_results:
        print("SIMULATING")
        gt = simulator.simulate(inflight_work[(work.partition, work.correlation_id)], model_cache, True)
        sim_results[(work.partition, work.correlation_id)] = gt
        
    gt = sim_results[(work.partition, work.correlation_id)]
    work.consistent_with(gt)
    if work.last_chunk:
        print("All chunks have been verified :)")

# Mount the frontend directory to serve static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
