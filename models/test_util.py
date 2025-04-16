from .cache import ModelCache, SafeTensorCache
from .graph import ComputeGraphEdge
from .tensor import Tensor
from .pipeline import CorrelatedTensor
from .queue import CorrelatedQueue
import numpy as np

# Helper functions
def random_2d_tensor(correlation_id: str, shape: tuple[int, int]) -> CorrelatedTensor:
    return CorrelatedTensor(correlation_id=correlation_id, tensor=Tensor.from_numpy(np.random.randn(*shape)))

def create_cq():
    """Creates and returns two edges and a CorrelatedQueue."""
    e0 = ComputeGraphEdge(src="a", src_output="0", dst="b", dst_input="0")
    e1 = ComputeGraphEdge(src="b", src_output="0", dst="c", dst_input="0")
    cq = CorrelatedQueue([e0, e1])
    return e0, e1, cq

def create_element(correlation_id: str, value: int) -> CorrelatedTensor:
    """Creates a PipelineElement with a single-element Tensor."""
    return CorrelatedTensor(correlation_id=correlation_id, tensor=Tensor(elements=[value], shape=[1]))

def llama_1b_cache() -> SafeTensorCache:
    return ModelCache().get_cache("meta-llama/Llama-3.2-1B")