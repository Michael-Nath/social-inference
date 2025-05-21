from dataclasses import dataclass
import torch
import numpy as np

from inference.encoding import read_be_int, read_encoded_string, size_encoded_string, write_be_int, write_encoded_string

# TODO: Handle dtypes
@dataclass
class Tensor:
    data: bytes
    shape: list[int]
    dtype: str = "float32"  # Default to float32

    @classmethod
    def from_numpy(cls, np_array: np.ndarray):
        return cls(
            data=np_array.tobytes(),
            shape=list(np_array.shape),
            dtype=str(np_array.dtype)
        )
    
    def to_numpy(self):
        return np.frombuffer(self.data, dtype=self.dtype).reshape(self.shape)
    
    @classmethod
    def from_torch(cls, torch_tensor):
        retyped = torch_tensor
        if retyped.dtype == torch.bfloat16:
            retyped = retyped.to(torch.float32)
        elif retyped.dtype == torch.float16:
            retyped = retyped.to(torch.float32)
        np_array = retyped.detach().cpu().numpy()
        return cls(
            data=np_array.tobytes(),
            shape=list(np_array.shape),
            dtype=str(np_array.dtype)
        )

    def to_torch(self):
        return torch.from_numpy(self.to_numpy())

@dataclass
class CorrelatedTensor:
    correlation_id: str
    tensor: Tensor

def read_encoded_tensor(offset: int, data: bytes) -> tuple[Tensor, int]:
    shapes_length, offset = read_be_int(offset, data)
    shapes = []
    for _ in range(shapes_length):
        shape, offset = read_be_int(offset, data)
        shapes.append(shape)
    dtype, offset = read_encoded_string(offset, data)
    data_length, offset = read_be_int(offset, data)
    data = data[offset:offset+data_length]
    return Tensor(shape=shapes, dtype=dtype, data=data), offset + data_length

def write_encoded_tensor(data: bytearray, offset: int, tensor: Tensor) -> int:
    assert isinstance(tensor, Tensor)

    offset = write_be_int(data, offset, len(tensor.shape))
    for shape in tensor.shape:
        offset = write_be_int(data, offset, shape)
    offset = write_encoded_string(data, offset, tensor.dtype)
    offset = write_be_int(data, offset, len(tensor.data))
    data[offset:offset+len(tensor.data)] = tensor.data
    return offset + len(tensor.data)

def size_encoded_tensor(tensor: Tensor) -> int:
    assert isinstance(tensor, Tensor)

    size = 4 # for the shapes length
    for shape in tensor.shape:
        size += 4 # for the shape
    size += size_encoded_string(tensor.dtype)
    size += 4 # for the data length
    size += len(tensor.data)
    return size

def read_encoded_correlated_tensor(offset: int, data: bytes) -> tuple[CorrelatedTensor, int]:
    correlation_id, offset = read_encoded_string(offset, data)
    tensor, offset = read_encoded_tensor(offset, data)
    return CorrelatedTensor(correlation_id=correlation_id, tensor=tensor), offset

def write_encoded_correlated_tensor(data: bytearray, offset: int, correlated_tensor: CorrelatedTensor) -> int:
    assert isinstance(correlated_tensor, CorrelatedTensor)

    offset = write_encoded_string(data, offset, correlated_tensor.correlation_id)
    offset = write_encoded_tensor(data, offset, correlated_tensor.tensor)
    return offset

def size_encoded_correlated_tensor(correlated_tensor: CorrelatedTensor) -> int:
    assert isinstance(correlated_tensor, CorrelatedTensor)

    return size_encoded_string(correlated_tensor.correlation_id) + size_encoded_tensor(correlated_tensor.tensor)
