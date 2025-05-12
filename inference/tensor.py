import base64
from pydantic import BaseModel, Field, field_serializer
import torch
import numpy as np


# TODO: Handle dtypes
class Tensor(BaseModel):
    elements: str
    shape: list[int]
    dtype: str = "float32"  # Default to float32

    @classmethod
    def from_numpy(cls, np_array: np.ndarray):
        return cls(
            elements=base64.b64encode(np_array.tobytes()).decode('ascii'),
            shape=list(np_array.shape),
            dtype=str(np_array.dtype)
        )
    
    def to_numpy(self):
        return np.frombuffer(base64.b64decode(self.elements), dtype=self.dtype).reshape(self.shape)
    
    @classmethod
    def from_torch(cls, torch_tensor):
        np_array = torch_tensor.detach().cpu().numpy()
        return cls(
            elements=base64.b64encode(np_array.tobytes()).decode('ascii'),
            shape=list(np_array.shape),
            dtype=str(np_array.dtype)
        )

    def to_torch(self):
        return torch.from_numpy(self.to_numpy())
    
class CorrelatedTensor(BaseModel):
    correlation_id: str
    tensor: Tensor