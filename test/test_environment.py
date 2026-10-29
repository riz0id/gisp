import pytest

from Evaluate import Environment, UnboundSymbol
from Syntax import Integer


def test_iteration_yields_the_frame_chain_innermost_first():
    outer = Environment({"a": Integer(1)})
    middle = Environment({"b": Integer(2)}, parent=outer)
    inner = Environment({"c": Integer(3)}, parent=middle)

    assert list(inner) == [inner, middle, outer]
    assert list(outer) == [outer]


def test_lookup_resolves_in_the_innermost_frame_that_binds_the_name():
    outer = Environment({"x": Integer(1)})
    inner = Environment({"x": Integer(2)}, parent=outer)

    assert inner.lookup("x") == Integer(2)
    assert outer.lookup("x") == Integer(1)


def test_assign_writes_the_innermost_binding_frame_only():
    outer = Environment({"x": Integer(1)})
    inner = Environment({"x": Integer(2)}, parent=outer)

    inner.assign("x", Integer(3))

    assert inner.bindings["x"] == Integer(3)
    assert outer.bindings["x"] == Integer(1)


def test_missing_names_still_raise_unbound_symbol():
    environment = Environment({"x": Integer(1)}, parent=Environment({}))

    with pytest.raises(UnboundSymbol):
        environment.lookup("y")
    with pytest.raises(UnboundSymbol):
        environment.assign("y", Integer(2))
