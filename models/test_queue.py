from queue import Queue
from .test_util import create_cq, create_element

def test_correlated_queue_happy_path():
    edge0, edge1, cq = create_cq()
    element1 = create_element("1", 1)
    element2 = create_element("1", 2)

    cq.put(edge0, element1)
    cq.put(edge1, element2)

    assert cq.pop(blocking=True) == {edge0: element1, edge1: element2}
    assert cq.pop(blocking=False) is None

def test_correlated_queue_timing():
    edge0, edge1, cq = create_cq()
    element1 = create_element("1", 1)
    element2 = create_element("1", 2)

    cq.put(edge0, element1)
    assert cq.pop(blocking=False) is None

    cq.put(edge1, element2)
    assert cq.pop(blocking=True) == {edge0: element1, edge1: element2}
    assert cq.pop(blocking=False) is None

def test_correlated_queue_out_of_order():
    e0, e1, cq = create_cq()
    e1_id1 = create_element("1", 1)
    e2_id2 = create_element("2", 2)
    e1_id2 = create_element("2", 3)
    e2_id1 = create_element("1", 3) # Corresponds to the second element for ID 1

    cq.put(e0, e1_id1)
    cq.put(e0, e2_id2)
    assert cq.pop(blocking=False) is None

    cq.put(e1, e1_id2) # Element for ID 2 into q1
    cq.put(e1, e2_id1) # Element for ID 1 into q2

    # Expected order based on correlation ID and queue source:
    expected1 = {e0: e1_id1, e1: e2_id1} # Elements for ID 1 (from q0 then q1)
    expected2 = {e0: e2_id2, e1: e1_id2} # Elements for ID 2 (from q0 then q1)

    assert cq.pop(blocking=True) == expected1
    assert cq.pop(blocking=True) == expected2
    assert cq.pop(blocking=False) is None
