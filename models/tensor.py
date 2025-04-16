from pydantic import BaseModel, Field
import torch
import numpy as np


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
    
class CorrelatedTensor(BaseModel):
    correlation_id: int
    tensor: Tensor