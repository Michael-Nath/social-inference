import torch
import time
import torch.fx.proxy
import torch.nn as nn
import torch._dynamo as dynamo
import inspect
import types
from transformers import AutoModelForCausalLM, AutoConfig, AutoModel
from accelerate import init_empty_weights

# Instead of making nontrivial dummy inputs, we set every argument to None.
def build_dummy_args(sig):
    dummy_args = {}
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            dummy_args[name] = None
    # If both 'input_ids' and 'inputs_embeds' are present, to satisfy the model's exclusive check,
    # we override 'input_ids' to be a dummy tensor.
    if "input_ids" in dummy_args and "inputs_embeds" in dummy_args:
         dummy_args["input_ids"] = torch.randint(0, 1000, (1, 16), device='meta')
    return dummy_args

def trace_model(core):
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

# Wrap the inner model so that its dummy arguments are passed as an explicit input.
def trace_inner_model(model):
    core = model.model  # target the inner model (avoids the wrapper and decorators)
    return trace_model(core) 

# Load the HuggingFace model.
# config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
# with init_empty_weights(include_buffers=True):
#     model = AutoModel.from_config(config)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
# Trace the inner forward function.
exported_module, dummy_args = trace_model(model)
# Now the exported module expects one argument (a dict of dummy inputs).
result = exported_module(dummy_args)

def decompose(module: torch.fx.GraphModule):
    new_graph = torch.fx.Graph()
    env = {}
    tracer = torch.fx.proxy.GraphAppendingTracer(new_graph)
    graph = module.graph
    m = dict(module.named_modules())
    for node in graph.nodes:
        if node.op == 'call_module':
            proxy_args = [torch.fx.Proxy(env[x.name], tracer) if isinstance(x, torch.fx.Node) else x for x in node.args]
            # breakpoint()
            m_ = m[node.target]
            output_proxy = m_(*proxy_args)
            new_node = output_proxy.node
            env[node.name] = new_node
        else:
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
    return torch.fx.GraphModule(module, new_graph)



g = decompose(exported_module)
targets = set()
for node in g.graph.nodes:
    if callable(node.target):
        #print(f"{node.name} is callable: {node.target}({node.args}, {node.kwargs})")
        targets.add(f"{node.op} : {node.target.__name__}({node.args}, {node.kwargs})")
    else:
        targets.add(f"{node.op} : {node.target}({node.args}, {node.kwargs})")
for t in targets:
    print(t)
# g.graph.print_tabular()
# graph = exported_module.graph
# print(g.code)