from contextlib import contextmanager
from typing import List

class NameScope:
    """
    A class that helps create nested node names using a stack-based scope system.
    """
    _scopes: List[str] = []

    @classmethod
    @contextmanager
    def push_scope(cls, scope_name: str):
        """
        Push a new scope onto the stack. Returns a context manager that automatically
        pops the scope when the context is exited.
        """
        cls._scopes.append(scope_name)
        try:
            yield
        finally:
            cls._scopes.pop()

    @classmethod
    def name(cls, name: str) -> str:
        """
        Create a fully qualified name by joining all scopes with dots and appending
        the final name.
        """
        if not cls._scopes:
            return name
        return ".".join(cls._scopes) + "." + name 