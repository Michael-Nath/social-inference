from collections import defaultdict, deque
import threading

from pydantic import BaseModel

from .graph import ComputeGraphEdge
from .tensor import Tensor

class CorrelatedTensor(BaseModel):
    """
    A correlated tensor is a tensor with a correlation ID

    Inherits from pydantic.BaseModel to allow for easy serialization
    """
    correlation_id: str
    tensor: Tensor

class CorrelatedQueue:
    """
    Queue that merges multiple input queues into a single output queue,
    reordering and blocking as necessary to maintain correlation.

    For example, if we have a partition with two inputs, then a correlated queue
    will ensure that when the partition will only dispatch work when both inputs
    are available. When it does dispatch work, the correlated queue will ensure
    that both PipelineElements popped are correlated (same correlation ID).

    CorrelatedQueue is MPMC thread-safe.

    Credit where credit is due: gemini-2.5-pro 3-shot this w/ tests provided by me ;)
    """


    def __init__(self, edges: list[ComputeGraphEdge]):
        """
        Initializes the CorrelatedQueue.

        Args:
            edges: A list of edges (ComputeGraphEdge).
        """
        # self.inputs: dict[ComputeGraphEdge, Queue] = {edge: Queue() for edge in edges}
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        # Stores partially collected elements for each correlation ID
        # Key: correlation_id (str)
        # Value: dict[ComputeGraphEdge, PipelineElement | None], where key corresponds to edge index
        self.pending: dict[str, dict[ComputeGraphEdge, CorrelatedTensor | None]] = defaultdict(lambda: {edge: None for edge in edges})
        # Stores correlation IDs that are complete
        self.completed: set[str] = set()
        # Stores correlation IDs in the order their first element arrived
        self.arrival_order: deque[str] = deque()
        # Store the expected edges for completion check
        self.expected_edges: set[ComputeGraphEdge] = set(edges)

    def put(self, edge: ComputeGraphEdge, element: CorrelatedTensor):
        """
        Adds an element associated with a specific edge.
        Updates internal state and notifies waiting consumers if a set becomes complete.

        Args:
            edge: The edge to add the element to.
            element: The element to add.
        """
        if edge not in self.expected_edges:
            raise ValueError(f"Edge {edge} is not part of this CorrelatedQueue's expected inputs.")

        corr_id = element.correlation_id
        with self.cond:
            # Track arrival order only if it's the first element seen for this ID
            if corr_id not in self.pending:
                # Initialize pending dict for this ID; defaultdict handles this implicitly on first access,
                # but we need to ensure arrival order is tracked *before* first element is added.
                self.pending[corr_id] # Ensure entry exists
                self.arrival_order.append(corr_id)

            # Store the element
            self.pending[corr_id][edge] = element

            # Check if this correlation ID is now complete
            current_items = self.pending[corr_id]
            is_complete = len(current_items) == len(self.expected_edges) and \
                          all(e in current_items and current_items[e] is not None for e in self.expected_edges)

            if is_complete and corr_id not in self.completed:
                self.completed.add(corr_id)
                # Notify potentially waiting threads that a set might be ready
                self.cond.notify()

    def pop(self, blocking: bool = True, timeout: float | None = None) -> dict[ComputeGraphEdge, CorrelatedTensor] | None:
        """
        Gets the next fully correlated set of PipelineElements.

        Args:
            blocking: If True, block until a correlated set is available or timeout occurs.
                      If False, return None immediately if no set is ready.
            timeout: Optional timeout in seconds for blocking mode.

        Returns:
            A dict mapping edges to PipelineElements with the same correlation ID, one from each
            input queue, or None if non-blocking and no set is ready, or if timeout occurs.
        """
        with self.cond:
            # Outer loop: Wait/poll until a returnable item according to arrival order is found
            while True:
                # Check if any already completed ID is at the head of arrival_order
                if self.arrival_order:
                    head_id = self.arrival_order[0]
                    if head_id in self.completed:
                        self.completed.remove(head_id)
                        self.arrival_order.popleft()
                        result = self.pending.pop(head_id)
                        # print(result)
                        # assert all(item is not None for item in result.values())
                        return result # Return the completed set

                # If the head of arrival order wasn't ready, decide whether to block or return
                if not blocking:
                    return None

                # If blocking, wait for a notification or timeout
                signaled = self.cond.wait(timeout=timeout)
                # After waiting or timeout, loop again to check status
                if not signaled and not (self.arrival_order and self.arrival_order[0] in self.completed):
                    # Timeout occurred and head is still not ready
                    return None
                # If signaled or wait completed without timeout, loop continues to re-check head_id
