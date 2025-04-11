from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from models import ModelCache
from models.model import Registration, RegistrationRequest, Work, WorkResult

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

@app.post("/register", response_model=Registration)
async def register(request: RegistrationRequest):
    """
    Called by clients to register their capabilities and request assignment to work.
    """

    return model_cache.register(request)
    
@app.get("/work", response_model=Work)
async def get_work():
    """
    Called by clients to request inference inputs
    """
    return model_cache.get_work()

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
        t = t.float()[:128, :128]
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
    Called by clients to submit inference results
    """
    return model_cache.submit_work(work)

# Mount the frontend directory to serve static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)