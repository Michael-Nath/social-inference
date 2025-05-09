from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig
import torch

def load_model(path: str):
  model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.float16,
    device_map="auto"
  )
  return model