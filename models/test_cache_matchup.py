import torch
from .utils import load_model
from inference.test_util import llama_1b_cache


def test_matchup():
  MODEL_PATH = "meta-llama/Llama-3.2-1B"
  model = load_model(MODEL_PATH)
  model_cache = llama_1b_cache()
  
  
  tensor_cache = model_cache.get_cache(MODEL_PATH)
  for name, param in model.named_parameters():
    param = param.cpu()
    if name == "model.embed_tokens.weight":
      continue
    with tensor_cache.get_tensor(name) as tensor:
      tensor = tensor.float().cpu()
      if len(tensor.shape) == 2:
        assert torch.allclose(tensor.T, param), breakpoint()
      else:
        assert torch.allclose(tensor, param), breakpoint()
    print(f"{name} matchup completed!")