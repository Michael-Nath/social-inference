from inference import ComputeGraphBuilder, ComputeGraph
# from utils import load_model

def simple_matmul() -> ComputeGraph:
    builder = ComputeGraphBuilder()
    x = builder.input("x")
    y = builder.input("y")
    with builder.partition("p0"):
        z = builder.matmul("z", x, y)
    builder.output("o", z)
    return builder.build()

# model = load_model("meta-llama/Llama-3.2-1B")
# params = list(model.model.named_parameters())
# param_keys = [p[0] for p in params]
# # print(param_keys)

# params = list(model.model.layers[0].named_parameters())
# param_keys = [p[0] for p in params]
# print(param_keys)
