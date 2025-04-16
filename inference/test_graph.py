import pytest
from .graph import ComputeGraphBuilder, ComputeGraph, ComputeGraphEdge, MatmulNode, DEFAULT_NODE_OUTPUT, InputNode, OutputNode, PARTITION_INPUT

def test_graph_construction():
    g = ComputeGraphBuilder()

    x = g.input("x")
    y = g.input("y")
    with g.partition("p0"):
        z = g.matmul("z", x, y)
    o = g.output("o", z)

    g = g.build()

    # Exhaustively test getters - only time we do this
    assert g.list_partition(PARTITION_INPUT) == {x.name, y.name}
    assert g.list_partition("p0") == {z.name}
    assert g.get_node(x.name) == x
    assert g.get_node(y.name) == y
    assert g.get_node(z.name) == z
    assert g.get_node(o.name) == o
    assert g.list_nodes() == {x.name, y.name, z.name, o.name}
    assert g.get_edges() == {
        ComputeGraphEdge(src=x.name, src_output=DEFAULT_NODE_OUTPUT, dst=z.name, dst_input=MatmulNode.LHS),
        ComputeGraphEdge(src=y.name, src_output=DEFAULT_NODE_OUTPUT, dst=z.name, dst_input=MatmulNode.RHS),
        ComputeGraphEdge(src=z.name, src_output=DEFAULT_NODE_OUTPUT, dst=o.name, dst_input=OutputNode.INPUT),
    }

def test_graph_partition():
    builder = ComputeGraphBuilder()

    x = builder.input("x")
    y = builder.input("y")
    with builder.partition("p0"):
        z = builder.matmul("z", x, y)
    o = builder.output("o", z)

    g = builder.build()

    p0 = g.extract_partition("p0")
    assert p0.get_node("z") == z
    assert p0.list_partition("p0") == {z.name}
    assert p0.get_forward_edges("x") == set()
    assert p0.get_forward_edges("y") == set()
    assert p0.get_forward_edges("z") == set()
    assert p0.get_backward_edges("z") == set()

    p0_cut = g.extract_partition("p0", include_cut_edges=True)
    assert p0_cut.get_node("z") == z
    assert p0_cut.list_partition("p0") == {z.name}
    assert p0_cut.get_forward_edges("x") == {ComputeGraphEdge(src="x", src_output=DEFAULT_NODE_OUTPUT, dst="z", dst_input=MatmulNode.LHS)}
    assert p0_cut.get_forward_edges("y") == {ComputeGraphEdge(src="y", src_output=DEFAULT_NODE_OUTPUT, dst="z", dst_input=MatmulNode.RHS)}
    assert p0_cut.get_forward_edges("z") == {ComputeGraphEdge(src="z", src_output=DEFAULT_NODE_OUTPUT, dst="o", dst_input=OutputNode.INPUT)}
    assert p0_cut.get_backward_edges("z") == {
        ComputeGraphEdge(src="y", src_output=DEFAULT_NODE_OUTPUT, dst="z", dst_input=MatmulNode.RHS),
        ComputeGraphEdge(src="x", src_output=DEFAULT_NODE_OUTPUT, dst="z", dst_input=MatmulNode.LHS),
    }


def test_graph_cuts():
    builder = ComputeGraphBuilder()

    x = builder.input("x")
    y = builder.input("y")
    with builder.partition("p0"):
        z = builder.matmul("z", x, y)
    with builder.partition("p1"):
        w = builder.matmul("w", z, z)
    o = builder.output("o", w)

    g = builder.build()

    assert g.identify_forward_cuts("p0") == {
        ComputeGraphEdge(src="x", src_output=DEFAULT_NODE_OUTPUT, dst="z", dst_input=MatmulNode.LHS),
        ComputeGraphEdge(src="y", src_output=DEFAULT_NODE_OUTPUT, dst="z", dst_input=MatmulNode.RHS),
    }
    assert g.identify_forward_cuts("p1") == {
        ComputeGraphEdge(src="z", src_output=DEFAULT_NODE_OUTPUT, dst="w", dst_input=MatmulNode.LHS),
        ComputeGraphEdge(src="z", src_output=DEFAULT_NODE_OUTPUT, dst="w", dst_input=MatmulNode.RHS),
    }
    assert g.identify_backward_cuts("p0") == {
        ComputeGraphEdge(src="z", src_output=DEFAULT_NODE_OUTPUT, dst="w", dst_input=MatmulNode.LHS),
        ComputeGraphEdge(src="z", src_output=DEFAULT_NODE_OUTPUT, dst="w", dst_input=MatmulNode.RHS),
    }
    assert g.identify_backward_cuts("p1") == {
        ComputeGraphEdge(src="w", src_output=DEFAULT_NODE_OUTPUT, dst="o", dst_input=OutputNode.INPUT),
    }