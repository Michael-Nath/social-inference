import torch
import torch.nn as nn
import torch.fx.proxy
import torch._dynamo as dynamo
import inspect
from transformers import AutoModel, AutoConfig
from accelerate import init_empty_weights
import contextlib

torch.autocast = lambda *args, **kwargs: contextlib.nullcontext()


def build_dummy_args(sig, keep_keys=("input_ids",)):
    dummy_args = {}
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            if name in keep_keys:
                dummy_args[name] = torch.randint(0, 1000, (1, 16), device='meta')
            else:
                dummy_args[name] = None
    if "input_ids" in dummy_args and "inputs_embeds" in dummy_args:
        dummy_args["inputs_embeds"] = None
    return dummy_args


def trace_model(core, keep_keys=("input_ids",)):
    raw_forward = core.forward
    sig = inspect.signature(raw_forward)
    dummy_args = build_dummy_args(sig, keep_keys)

    class CoreWrapper(nn.Module):
        def __init__(self, raw_forward):
            super().__init__()
            self.raw_forward = raw_forward
        def forward(self, **kwargs):
            return self.raw_forward(**kwargs)

    core_module = CoreWrapper(raw_forward)
    exported_module = dynamo.export(core_module, **dummy_args)[0]
    return exported_module, dummy_args


def extract_subgraph_dependent_on_input(module: torch.fx.GraphModule, input_name: str):
    graph = module.graph
    dependencies = {}

    for node in reversed(graph.nodes):
        depends = False
        if node.op == "placeholder" and node.name == input_name:
            depends = True
        else:
            for arg in node.all_input_nodes:
                if dependencies.get(arg.name, False):
                    depends = True
                    break
        dependencies[node.name] = depends

    new_graph = torch.fx.Graph()
    env = {}
    for node in graph.nodes:
        if dependencies.get(node.name, False):
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node

    return torch.fx.GraphModule(module, new_graph)


def trace_inner_model(model, keep_keys=("input_ids",)):
    core = model.model
    return trace_model(core, keep_keys)


config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-405B-FP8")
with init_empty_weights():
    model = AutoModel.from_config(config)

exported_module, dummy_args = trace_model(model, keep_keys=("input_ids",))
filtered_module = extract_subgraph_dependent_on_input(exported_module, "l_kwargs_input_ids_")

filtered_module.graph.print_tabular()
# exported_module.graph.print_tabular()
