from inference import ComputeGraphBuilder, ComputeGraph

def simple_matmul() -> ComputeGraph:
    builder = ComputeGraphBuilder()
    x = builder.input("x")
    y = builder.input("y")
    with builder.partition("p0"):
        z = builder.matmul("z", x, y)
    builder.output("o", z)
    return builder.build()