import torch
import time
import torch.nn as nn
import torch._dynamo as dynamo
import inspect
import types
from transformers import AutoModelForCausalLM

# Instead of making nontrivial dummy inputs, we set every argument to None.
def build_dummy_args(sig):
    dummy_args = {}
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            dummy_args[name] = None
    # If both 'input_ids' and 'inputs_embeds' are present, to satisfy the model's exclusive check,
    # we override 'input_ids' to be a dummy tensor.
    if "input_ids" in dummy_args and "inputs_embeds" in dummy_args:
         dummy_args["input_ids"] = torch.randint(0, 1000, (1, 16))
    return dummy_args

# Wrap the inner model so that its dummy arguments are passed as an explicit input.
def trace_inner_model(model):
    core = model.model  # target the inner model (avoids the wrapper and decorators)
    raw_forward = core.forward
    sig = inspect.signature(raw_forward)
    dummy_args = build_dummy_args(sig)
    
    # Define a wrapper module whose forward takes the dummy arguments as an explicit input.
    class CoreWrapper(nn.Module):
        def __init__(self, raw_forward):
            super().__init__()
            self.raw_forward = raw_forward
        def forward(self, input_args):
            return self.raw_forward(**input_args)
    
    core_module = CoreWrapper(raw_forward)
    # Pass dummy_args so that Dynamo sees them as an explicit input.
    exported_module = dynamo.export(core_module, dummy_args)[0]
    return exported_module, dummy_args

# Load the HuggingFace model.
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model.eval()

# Trace the inner forward function.
exported_module, dummy_args = trace_inner_model(model)
# Now the exported module expects one argument (a dict of dummy inputs).
result = exported_module(dummy_args)




graph = exported_module.graph
print(exported_module.code)