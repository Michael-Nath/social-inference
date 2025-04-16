import pytest
from .graph import ComputeGraph, ComputeGraphEdge, MatmulNode, DEFAULT_NODE_OUTPUT, InputNode, OutputNode

def test_graph_construction():
    g = ComputeGraph()

    x = g.input("x")
    y = g.input("y")
    with g.partition("p0"):
        z = g.matmul("z", x, y)
    o = g.output("o", z)

    assert g.nodes == {
        "x": x,
        "y": y,
        "z": z,
        "o": o,
    }

def test_graph_partition():
    g = ComputeGraph()

    x = g.input("x")
    y = g.input("y")
    with g.partition("p0"):
        z = g.matmul("z", x, y)
    g.output("o", z)

    p0 = g.extract_partition("p0")
    assert p0.nodes["z"] == z
    assert p0.partitions["p0"] == {"z"}
    assert p0.get_forward_edges("x") == set()
    assert p0.get_forward_edges("y") == set()
    assert p0.get_forward_edges("z") == set()
    assert p0.get_backward_edges("z") == set()

    p0_cut = g.extract_partition("p0", include_cut_edges=True)
    assert p0_cut.nodes["z"] == z
    assert p0_cut.partitions["p0"] == {"z"}
    assert p0_cut.get_forward_edges("x") == {ComputeGraphEdge(src="x", src_output=DEFAULT_NODE_OUTPUT, dst="z", dst_input=MatmulNode.LHS)}
    assert p0_cut.get_forward_edges("y") == {ComputeGraphEdge(src="y", src_output=DEFAULT_NODE_OUTPUT, dst="z", dst_input=MatmulNode.RHS)}
    assert p0_cut.get_forward_edges("z") == {ComputeGraphEdge(src="z", src_output=DEFAULT_NODE_OUTPUT, dst="o", dst_input=OutputNode.INPUT)}
    assert p0_cut.get_backward_edges("z") == {
        ComputeGraphEdge(src="y", src_output=DEFAULT_NODE_OUTPUT, dst="z", dst_input=MatmulNode.RHS),
        ComputeGraphEdge(src="x", src_output=DEFAULT_NODE_OUTPUT, dst="z", dst_input=MatmulNode.LHS),
    }


def test_graph_cuts():
    g = ComputeGraph()

    x = g.input("x")
    y = g.input("y")
    with g.partition("p0"):
        z = g.matmul("z", x, y)
    with g.partition("p1"):
        w = g.matmul("w", z, z)
    o = g.output("o", w)

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

def test_graph_freeze():
    g = ComputeGraph()
    g.input("x")
    assert not g.is_frozen()
    g.freeze()
    assert g.is_frozen()

    # Check that the graph is frozen
    with pytest.raises(ValueError):
        g.input("y")
