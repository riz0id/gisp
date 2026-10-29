import pytest
from hypothesis import given
from hypothesis import strategies as st

from Resolve import Ambiguous, BindingTable, Unbound, bindings, resolve, same_variable
from Syntax import Expr, Scope, SList, Symbol

# (lambda (x{1}) (lambda (x{1,2}) x{1,2}))
#
# The scope sets are written out by hand so the expectations below are exact:
# the inner binder carries the outer form's scope plus its own, and the body
# occurrence carries everything it sits inside.
outer = Symbol("x", {Scope(1)})
inner = Symbol("x", {Scope(1), Scope(2)})
occurrence = Symbol("x", {Scope(1), Scope(2)})

program = SList(
    [
        Symbol("lambda"),
        SList([outer]),
        SList([Symbol("lambda"), SList([inner]), occurrence]),
    ]
)

table = bindings(program)


def test_the_inner_binder_shadows_the_outer():
    assert resolve(occurrence, table).binder == inner
    assert inner != outer


def test_resolution_returns_the_table_entry_for_the_binder():
    assert resolve(occurrence, table) == table[inner]
    assert table[inner].binder == inner


def test_incomparable_candidates_are_ambiguous():
    # Macro-introduced identifiers carry overlapping, non-nested scope sets:
    # here both binders are visible from the reference and neither is more
    # specific than the other, so no binding answers the reference.
    left = Symbol("x", {Scope(3), Scope(4)})
    right = Symbol("x", {Scope(3), Scope(6)})
    reference = Symbol("x", {Scope(3), Scope(4), Scope(6)})

    with pytest.raises(Ambiguous):
        resolve(reference, BindingTable([left, right]))


def test_a_free_spelling_is_unbound():
    with pytest.raises(Unbound):
        resolve(Symbol("y", {Scope(1), Scope(2)}), table)


def test_occurrences_of_one_binding_are_the_same_variable():
    assert same_variable(occurrence, Symbol("x", {Scope(1), Scope(2)}), table)
    assert not same_variable(occurrence, Symbol("x", {Scope(1)}), table)


def test_resolution_is_a_function_of_the_symbol_alone():
    before = resolve(occurrence, table)
    embedded = SList([Symbol("f", {Scope(1)}), program, Symbol("g")])
    after = resolve(occurrence, bindings(embedded))
    assert before == after


def test_a_binder_resolves_to_its_own_binding():
    assert resolve(inner, table).binder == inner
    assert resolve(outer, table).binder == outer


def test_bindings_collects_letrec_binders():
    # (letrec ((f{5} g{5}) (g{5} f{5})) f{5}) — both names are binding
    # sites carrying the form's one scope, and each initializer reference
    # resolves through them, so the walk must record the names and nothing
    # else.
    f_binder = Symbol("f", {Scope(5)})
    g_binder = Symbol("g", {Scope(5)})
    form = SList(
        [
            Symbol("letrec"),
            SList(
                [
                    SList([f_binder, Symbol("g", {Scope(5)})]),
                    SList([g_binder, Symbol("f", {Scope(5)})]),
                ]
            ),
            Symbol("f", {Scope(5)}),
        ]
    )
    letrec_table = bindings(form)

    assert set(letrec_table) == {f_binder, g_binder}
    assert resolve(Symbol("g", {Scope(5)}), letrec_table).binder == g_binder
    assert resolve(Symbol("f", {Scope(5)}), letrec_table).binder == f_binder


spellings = st.sampled_from(["x", "y", "z"])


@given(st.lists(spellings, min_size=1, max_size=8), spellings)
def test_the_deepest_binder_with_the_spelling_wins(names, ref_name):
    # A chain of single-parameter lambdas: the binder at depth d carries the
    # scopes of every enclosing form plus its own, and the innermost body
    # reference carries all of them, so every binder is a candidate for its
    # own spelling and subset ordering picks the deepest.
    depth = len(names)
    binders = [
        Symbol(name, {Scope(i) for i in range(d + 1)}) for d, name in enumerate(names)
    ]
    ref = Symbol(ref_name, {Scope(i) for i in range(depth)})

    body: Expr = ref
    for binder in reversed(binders):
        body = SList([Symbol("lambda"), SList([binder]), body])

    chain_table = bindings(body)

    bound_depths = [d for d, name in enumerate(names) if name == ref_name]
    if not bound_depths:
        with pytest.raises(Unbound):
            resolve(ref, chain_table)
    else:
        # resolve raising Ambiguous would fail this test directly, so the
        # assertion below also witnesses that Ambiguous is never reached.
        assert resolve(ref, chain_table).binder == binders[max(bound_depths)]
