import pytest
from .graph import ComputeGraph, ComputeGraphEdge

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
        ComputeGraphEdge(src="x", src_output="output", dst="z", dst_input="lhs"),
        ComputeGraphEdge(src="y", src_output="output", dst="z", dst_input="rhs"),
    }
    assert g.identify_forward_cuts("p1") == {
        ComputeGraphEdge(src="z", src_output="output", dst="w", dst_input="lhs"),
        ComputeGraphEdge(src="z", src_output="output", dst="w", dst_input="rhs"),
    }
    assert g.identify_backward_cuts("p0") == {
        ComputeGraphEdge(src="z", src_output="output", dst="w", dst_input="lhs"),
        ComputeGraphEdge(src="z", src_output="output", dst="w", dst_input="rhs"),
    }
    assert g.identify_backward_cuts("p1") == {
        ComputeGraphEdge(src="w", src_output="output", dst="o", dst_input="input"),
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
