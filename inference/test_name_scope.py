import pytest
from inference.name_scope import NameScope

def test_basic_scope():
    """Test basic scope functionality"""
    with NameScope.push_scope("test"):
        assert NameScope.name("node") == "test.node"

def test_nested_scopes():
    """Test nested scope functionality"""
    with NameScope.push_scope("outer"):
        with NameScope.push_scope("middle"):
            with NameScope.push_scope("inner"):
                assert NameScope.name("node") == "outer.middle.inner.node"
            assert NameScope.name("node") == "outer.middle.node"
        assert NameScope.name("node") == "outer.node"
    assert NameScope.name("node") == "node"

def test_empty_scope():
    """Test behavior when no scopes are active"""
    assert NameScope.name("node") == "node"

def test_scope_cleanup():
    """Test that scopes are properly cleaned up after context exit"""
    with NameScope.push_scope("test"):
        pass
    assert NameScope.name("node") == "node"

def test_scope_cleanup_on_exception():
    """Test that scopes are properly cleaned up even when exceptions occur"""
    try:
        with NameScope.push_scope("test"):
            raise ValueError("Test exception")
    except ValueError:
        pass
    assert NameScope.name("node") == "node"

def test_multiple_scopes():
    """Test multiple independent scope blocks"""
    with NameScope.push_scope("first"):
        assert NameScope.name("node") == "first.node"
    
    with NameScope.push_scope("second"):
        assert NameScope.name("node") == "second.node"
    
    assert NameScope.name("node") == "node" 