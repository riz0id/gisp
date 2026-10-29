from collections.abc import Callable
from itertools import count
from types import MappingProxyType
from typing import Final

import pytest

from Expander import ParseError, expand, parse
from Reader import read
from Resolve import (
    Ambiguous,
    Binding,
    BindingTable,
    Phase,
    Unbound,
    bindings,
    resolve,
    same_variable,
)
from Syntax import Integer, Scope, SList, Symbol

# Fixtures inject deterministic counters from the 1500s so no expectation
# depends on the global fresh-uid counter and no uid collides with another
# test file's hand-written scopes.

# Racket's name for phase 1: the phase a phase-0 form is expanded in.
TRANSFORMER: Final[Phase] = Phase(1)


def counter(start: int) -> Callable[[], Scope]:
    uids = count(start)
    return lambda: Scope(next(uids))


# -- the phase-0 default ----------------------------------------------------------


def test_every_member_defaults_to_phase_zero():
    binder = Symbol("x", {Scope(1500)})
    table = BindingTable([binder])

    assert Phase.runtime() == Phase(0)
    assert table.at(Phase.runtime()) == {binder: Binding(binder)}
    assert set(table) == {binder}
    assert len(table) == 1
    assert binder in table
    assert table[binder] == Binding(binder, Phase.runtime())
    assert resolve(binder, table) == Binding(binder)


def test_the_canonical_shadowing_fixture_still_resolves():
    # The fixture every binding expectation in the suite is written against:
    # indexing the table by phase must leave it exactly as it was.
    outer = Symbol("x", {Scope(1501)})
    inner = Symbol("x", {Scope(1501), Scope(1502)})
    program = SList(
        [
            Symbol("lambda"),
            SList([outer]),
            SList([Symbol("lambda"), SList([inner]), inner]),
        ]
    )
    table = bindings(program)

    assert resolve(inner, table).binder == inner
    assert resolve(outer, table).binder == outer


def test_the_default_table_is_the_base_environment():
    table = BindingTable.default()

    assert set(table) == {Symbol("#t"), Symbol("#f"), Symbol("prod#")}
    assert table.at(TRANSFORMER) == {}


# -- separation -------------------------------------------------------------------


def test_a_phase_one_binder_is_invisible_at_phase_zero():
    binder = Symbol("x", {Scope(1505)})
    table = BindingTable([binder], TRANSFORMER)

    assert resolve(binder, table, TRANSFORMER).binder == binder
    with pytest.raises(Unbound):
        resolve(binder, table)


def test_a_phase_zero_binder_is_invisible_at_phase_one():
    # The same reference and the same scope set in both directions, so the
    # phase and not the scopes is what decided each answer.
    binder = Symbol("x", {Scope(1506)})
    table = BindingTable([binder])

    assert resolve(binder, table).binder == binder
    with pytest.raises(Unbound):
        resolve(binder, table, TRANSFORMER)


def test_one_spelling_bound_in_two_layers_is_two_bindings():
    runtime_binder = Symbol("x", {Scope(1507)})
    transformer_binder = Symbol("x", {Scope(1508)})
    table = BindingTable()
    table.register(runtime_binder)
    table.register(transformer_binder, TRANSFORMER)

    # Both binders are visible from this reference, and each layer holds
    # exactly one of them.
    reference = Symbol("x", {Scope(1507), Scope(1508)})

    assert resolve(reference, table).binder == runtime_binder
    assert resolve(reference, table, TRANSFORMER).binder == transformer_binder
    assert resolve(reference, table) != resolve(reference, table, TRANSFORMER)


def test_the_phase_distinguishes_two_bindings_of_one_binder():
    binder = Symbol("x", {Scope(1509)})
    table = BindingTable()

    assert table.register(binder) == Binding(binder, Phase.runtime())
    assert table.register(binder, TRANSFORMER) == Binding(binder, TRANSFORMER)
    assert table.register(binder) != table.register(binder, TRANSFORMER)


def test_same_variable_answers_at_its_phase():
    binder = Symbol("x", {Scope(1510)})
    reference = Symbol("x", {Scope(1510), Scope(1511)})
    table = BindingTable([binder], TRANSFORMER)

    assert same_variable(binder, reference, table, TRANSFORMER)
    with pytest.raises(Unbound):
        same_variable(binder, reference, table)


# -- layers -----------------------------------------------------------------------


def test_an_absent_layer_reads_as_empty_and_is_not_created():
    table = BindingTable([Symbol("x", {Scope(1512)})])

    assert table.at(TRANSFORMER) == {}
    assert TRANSFORMER not in table.binders


def test_a_layer_is_a_read_only_view():
    # `register` is the only writer, so a layer handed out cannot be added to
    # behind the table's back.
    binder = Symbol("x", {Scope(1513)})
    table = BindingTable([binder])
    layer = table.at(Phase.runtime())

    assert isinstance(layer, MappingProxyType)
    assert layer == {binder: Binding(binder)}


def test_ambiguity_is_per_layer():
    # Both binders are visible from the reference and neither is more specific
    # than the other, so one layer holding both answers nothing — while the
    # same two split across layers leave one candidate in each.
    left = Symbol("x", {Scope(1515), Scope(1516)})
    right = Symbol("x", {Scope(1515), Scope(1517)})
    reference = Symbol("x", {Scope(1515), Scope(1516), Scope(1517)})

    with pytest.raises(Ambiguous):
        resolve(reference, BindingTable([left, right]))

    split = BindingTable([left])
    split.register(right, TRANSFORMER)

    assert resolve(reference, split).binder == left
    assert resolve(reference, split, TRANSFORMER).binder == right


# -- expansion --------------------------------------------------------------------


def test_expanding_at_phase_one_produces_the_same_tree():
    # Scopes are phase-independent: a binding form's scope applies at every
    # phase, and only the layer a binder lands in differs. This is the property
    # that licenses keeping one scope set per symbol rather than one per phase.
    source = "(letrec ((f (lambda (x) (f x)))) (f 1))"
    at_runtime = expand(read(source), fresh=counter(1520))
    at_transformer = expand(read(source), fresh=counter(1520), phase=TRANSFORMER)

    assert at_runtime == at_transformer


def test_bindings_reconstructs_into_the_layer_it_is_given():
    expanded = expand(read("(lambda (x) x)"), fresh=counter(1525))
    table = bindings(expanded, TRANSFORMER)
    binder = Symbol("x", {Scope(1525)})

    assert resolve(binder, table, TRANSFORMER).binder == binder
    assert table.at(Phase.runtime()) == {}
    with pytest.raises(Unbound):
        resolve(binder, table)


def test_letrec_syntax_expands_at_phase_one():
    # The macro binder is registered in this expansion's layer, and the
    # expander reads the binding back out of that layer rather than the
    # phase-0 view.
    source = "(letrec-syntax ((m (syntax-rules () ((m a) a)))) (m 5))"

    assert expand(read(source), fresh=counter(1530)) == Integer(5)
    assert expand(read(source), fresh=counter(1530), phase=TRANSFORMER) == Integer(5)


# -- parse ------------------------------------------------------------------------


def test_parse_discharges_binders_at_its_phase():
    expanded = expand(read("(lambda (x) x)"), fresh=counter(1535))
    table = bindings(expanded, TRANSFORMER)
    binder = Symbol("x", {Scope(1535)})

    assert parse(expanded, table, TRANSFORMER) == SList(
        [Symbol("lambda"), SList([binder]), binder]
    )

    # A binder absent from the layer being read means the store does not
    # belong to this program, which is refused rather than fallen back from.
    with pytest.raises(ParseError):
        parse(expanded, table)


def test_a_reference_parses_to_its_spelling_outside_its_phase():
    reference = Symbol("x", {Scope(1537)})
    table = BindingTable([Symbol("x", {Scope(1537)})], TRANSFORMER)

    assert parse(reference, table, TRANSFORMER) == Symbol("x", {Scope(1537)})
    assert parse(reference, table) == Symbol("x")


def test_the_canonical_composition_holds_at_any_phase():
    expanded = expand(
        read("(letrec ((n 5)) n)"), fresh=counter(1540), phase=TRANSFORMER
    )
    parsed = parse(expanded, bindings(expanded, TRANSFORMER), TRANSFORMER)
    binder = Symbol("n", {Scope(1540)})

    assert parsed == SList(
        [Symbol("letrec"), SList([SList([binder, Integer(5)])]), binder]
    )
