from .cache import ModelCache, SafeTensorCache
from .graph import ComputeGraphEdge
from .tensor import Tensor
from .pipeline import CorrelatedTensor
from .queue import CorrelatedQueue
import numpy as np

# Helper functions
def random_2d_tensor(shape: tuple[int, int]) -> Tensor:
    return Tensor.from_numpy(np.random.randn(*shape))
def random_correlated_2d_tensor(correlation_id: str, shape: tuple[int, int]) -> CorrelatedTensor:
    return CorrelatedTensor(correlation_id=correlation_id, tensor=random_2d_tensor(shape))
def create_cq():
    """Creates and returns two edges and a CorrelatedQueue."""
    e0 = ComputeGraphEdge(src="a", src_output="0", dst="b", dst_input="0")
    e1 = ComputeGraphEdge(src="b", src_output="0", dst="c", dst_input="0")
    cq = CorrelatedQueue([e0, e1])
    return e0, e1, cq

def create_element(correlation_id: str, value: int) -> CorrelatedTensor:
    """Creates a PipelineElement with a single-element Tensor."""
    t = np.ones((1))
    return CorrelatedTensor(correlation_id=correlation_id, tensor=Tensor.from_numpy(t))

def llama_1b_cache() -> ModelCache:
    return ModelCache()