import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from Strategies import symbol_names, symbols

from Evaluate import Environment, EvalError, UnboundSymbol, evaluate
from Syntax import Boolean, Char, Integer, Nil, SList, Symbol

bounded_integers = st.integers(min_value=-10000, max_value=10000)

self_evaluating_atoms = st.one_of(
    bounded_integers.map(Integer),
    st.characters(exclude_categories=("Cs",)).map(lambda ch: Char(ord(ch))),
    st.booleans().map(Boolean),
    st.just(Nil()),
)


@given(self_evaluating_atoms)
def test_atoms_evaluate_to_themselves(atom):
    assert evaluate(atom, Environment({})) == atom


@given(symbols)
def test_an_unknown_symbol_is_unbound(name):
    with pytest.raises(UnboundSymbol):
        evaluate(name, Environment({}))


@given(st.dictionaries(symbol_names, self_evaluating_atoms))
def test_the_empty_combination_is_an_error(bindings):
    with pytest.raises(EvalError):
        evaluate(SList([]), Environment(bindings))


@given(self_evaluating_atoms, self_evaluating_atoms)
def test_if_evaluates_exactly_the_branch_selected_by_the_test(test, taken):
    untaken = Symbol("unbound")
    branches = [taken, untaken] if test == Boolean(True) else [untaken, taken]
    conditional = SList([Symbol("if"), Symbol("flag"), *branches])
    assert evaluate(conditional, Environment({"flag": test})) == taken


@given(symbol_names, self_evaluating_atoms, self_evaluating_atoms)
def test_letrec_bindings_shadow_the_environment_in_the_body(name, inner, outer):
    form = SList(
        [Symbol("letrec"), SList([SList([Symbol(name), inner])]), Symbol(name)]
    )
    assert evaluate(form, Environment({name: outer})) == inner


@given(symbols, symbols, self_evaluating_atoms)
def test_letrec_prebinds_names_to_nil_then_assigns_in_order(first, second, value):
    assume(first != second)
    forward = SList(
        [
            Symbol("letrec"),
            SList([SList([first, second]), SList([second, value])]),
            first,
        ]
    )
    backward = SList(
        [
            Symbol("letrec"),
            SList([SList([first, value]), SList([second, first])]),
            second,
        ]
    )
    assert evaluate(forward, Environment({})) == Nil()
    assert evaluate(backward, Environment({})) == value


@given(symbol_names, self_evaluating_atoms, self_evaluating_atoms)
def test_lambda_parameters_bind_arguments_and_shadow_the_environment(
    name, value, outer
):
    procedure = SList([Symbol("lambda"), SList([Symbol(name)]), Symbol(name)])
    assert evaluate(SList([procedure, value]), Environment({name: outer})) == value


@given(symbols, self_evaluating_atoms)
def test_lambda_captures_its_definition_environment(name, value):
    procedure = SList([Symbol("lambda"), SList([]), name])
    escape = SList([Symbol("letrec"), SList([SList([name, value])]), procedure])
    assert evaluate(SList([escape]), Environment({})) == value


# -- type ascription is erased at runtime ----------------------------------------


def ascribed(identifier: str, spelling: str = "@int") -> SList:
    return SList([Symbol("ascribe#"), Symbol(identifier), Symbol(spelling)])


@given(symbol_names, self_evaluating_atoms)
def test_an_ascribed_parameter_binds_like_a_bare_one(name, value):
    # Ascription carries no runtime meaning, so an ascribed lambda is the bare
    # lambda it erases to.
    ascribe = SList([Symbol("lambda"), SList([ascribed(name)]), Symbol(name)])
    bare = SList([Symbol("lambda"), SList([Symbol(name)]), Symbol(name)])

    assert evaluate(SList([ascribe, value]), Environment({})) == evaluate(
        SList([bare, value]), Environment({})
    )


@given(symbol_names, self_evaluating_atoms)
def test_an_ascribed_parameter_is_not_an_extra_parameter(name, value):
    # The type operand names no value, so erasing it must not leave a closure
    # of arity two that no call site can satisfy.
    procedure = SList([Symbol("lambda"), SList([ascribed(name)]), Symbol(name)])

    assert evaluate(SList([procedure, value]), Environment({})) == value


@given(symbol_names, bounded_integers)
def test_an_ascribed_letrec_binder_binds_like_a_bare_one(name, value):
    form = SList(
        [
            Symbol("letrec"),
            SList([SList([ascribed(name), Integer(value)])]),
            Symbol(name),
        ]
    )

    assert evaluate(form, Environment({})) == Integer(value)


@given(symbol_names, symbol_names, bounded_integers, bounded_integers)
def test_an_ascribed_letrec_binder_coexists_with_a_bare_sibling(
    first, second, left, right
):
    # The arm guard reads every binding, so an ascribed binder must not
    # disqualify the bare siblings beside it.
    assume(first != second)
    form = SList(
        [
            Symbol("letrec"),
            SList(
                [
                    SList([ascribed(first), Integer(left)]),
                    SList([Symbol(second), Integer(right)]),
                ]
            ),
            Symbol(second),
        ]
    )

    assert evaluate(form, Environment({})) == Integer(right)


@given(symbol_names, self_evaluating_atoms, self_evaluating_atoms)
def test_an_ascribed_parameter_still_shadows_the_environment(name, value, outer):
    procedure = SList([Symbol("lambda"), SList([ascribed(name)]), Symbol(name)])

    assert evaluate(SList([procedure, value]), Environment({name: outer})) == value
