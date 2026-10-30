import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from Strategies import symbol_names, symbols

from Evaluate import BUILTINS, Environment, EvalError, UnboundSymbol, evaluate
from Syntax import (
    Char,
    Integer,
    Residual,
    SList,
    Symbol,
    boolean,
    listed,
    unit,
)

bounded_integers = st.integers(min_value=-10000, max_value=10000)

# The two atoms that are their own value. A boolean and a product are
# constructions now, so neither is written as a datum and neither evaluates to
# itself.
self_evaluating_atoms = st.one_of(
    bounded_integers.map(Integer),
    st.characters(exclude_categories=("Cs",)).map(lambda ch: Char(ord(ch))),
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


@given(st.booleans(), self_evaluating_atoms)
def test_a_match_evaluates_exactly_the_clause_its_scrutinee_selects(decided, taken):
    # Booleans and not any atom, which is what the `if` this replaces took: a
    # `(#t (prod#))` pattern against a value of another sort is a type error
    # the checker refuses, and the evaluator refuses it too rather than quietly
    # falling through to the next clause.
    #
    # The clause not selected is unbound, so evaluating it would have failed.
    test = boolean(decided)
    untaken = Symbol("unbound")
    bodies = [taken, untaken] if decided else [untaken, taken]
    form = SList(
        [
            Symbol("match"),
            Symbol("flag"),
            SList([SList([Symbol("#t"), SList([Symbol("prod#")])]), bodies[0]]),
            SList([SList([Symbol("#f"), SList([Symbol("prod#")])]), bodies[1]]),
        ]
    )

    assert evaluate(form, Environment({"flag": test})) == taken


@given(self_evaluating_atoms)
def test_an_irrefutable_clause_binds_the_scrutinee(value):
    form = SList(
        [
            Symbol("match"),
            Symbol("flag"),
            SList([Symbol("bound"), Symbol("bound")]),
        ]
    )

    assert evaluate(form, Environment({"flag": value})) == value


def test_a_list_pattern_takes_a_list_apart():
    form = SList(
        [
            Symbol("match"),
            SList([Symbol("quote"), SList([Integer(1), Integer(2)])]),
            SList(
                [
                    SList([Symbol("nil"), SList([Symbol("prod#")])]),
                    Symbol("unbound"),
                ]
            ),
            SList(
                [
                    # The pattern and the body are the same text: the cell is
                    # taken apart by the construction that builds it, down to
                    # the `prod#` its payload is written with. Taking a list
                    # apart and putting it back together is what says so.
                    SList(
                        [
                            Symbol("cons"),
                            SList([Symbol("prod#"), Symbol("x"), Symbol("rest")]),
                        ]
                    ),
                    SList(
                        [
                            Symbol("cons"),
                            SList([Symbol("prod#"), Symbol("x"), Symbol("rest")]),
                        ]
                    ),
                ]
            ),
        ]
    )

    assert evaluate(form, Environment(BUILTINS)) == listed([Integer(1), Integer(2)])


def test_the_empty_list_matches_only_the_empty_list_pattern():
    def over(items: list[Expr]) -> Value:
        return evaluate(
            SList(
                [
                    Symbol("match"),
                    SList([Symbol("quote"), SList(items)]),
                    SList(
                        [
                            SList([Symbol("nil"), SList([Symbol("prod#")])]),
                            Integer(0),
                        ]
                    ),
                    SList(
                        [
                            SList(
                                [
                                    Symbol("cons"),
                                    SList(
                                        [
                                            Symbol("prod#"),
                                            Symbol("x"),
                                            Symbol("rest"),
                                        ]
                                    ),
                                ]
                            ),
                            Integer(1),
                        ]
                    ),
                ]
            ),
            Environment(BUILTINS),
        )

    assert over([]) == Integer(0)
    assert over([Integer(7)]) == Integer(1)


def construction(constructor: str) -> Value:
    return evaluate(
        SList([Symbol(constructor), SList([Symbol("prod#")])]),
        Environment(BUILTINS),
    )


def test_a_boolean_constructor_evaluates_to_a_boolean():
    # `@bool` is a sum, so a boolean is built the way a sum's values are: the
    # constructor applied to its payload. The result is a value and not a
    # residual — a constructor application is not stuck, it is already done.
    assert construction("#t") == boolean(True)
    assert construction("#f") == boolean(False)


def test_the_empty_product_constructor_evaluates_to_the_empty_tuple():
    assert evaluate(SList([Symbol("prod#")]), Environment(BUILTINS)) == unit()


def test_a_product_and_a_list_of_the_same_width_are_distinct_values():
    # What representing both as an `SList` made impossible: a two-field
    # product and a two-element list were the same value, and `eq` said so.
    # Now the constructor is what tells them apart.
    pair = SList([Symbol("prod#"), Integer(1), Integer(2)])
    both = SList([Symbol("quote"), SList([Integer(1), Integer(2)])])

    assert evaluate(pair, Environment(BUILTINS)) != evaluate(
        both, Environment(BUILTINS)
    )
    assert evaluate(
        SList([Symbol("eq"), pair, both]), Environment(BUILTINS)
    ) == boolean(False)


def test_a_boolean_is_distinct_from_the_product_it_carries():
    assert construction("#t") != unit()
    assert construction("#t") != construction("#f")


@given(st.booleans())
def test_a_constructed_boolean_is_the_one_a_comparison_answers_with(decided):
    # The property that matters: `@bool` has one representation however a
    # boolean was reached, so a built one and a compared one are the same value
    # and `eq` says so. Were they two representations, a `(#t (prod#))` pattern
    # would match one and not the other.
    built: Value = construction("#t" if decided else "#f")
    compared: Value = evaluate(
        SList([Symbol("le"), Integer(1), Integer(2 if decided else 0)]),
        Environment(BUILTINS),
    )

    assert built == compared


def test_the_empty_list_constructor_evaluates_to_the_empty_list():
    assert construction("nil") == listed([])


def test_a_cell_constructor_reads_the_payload_a_wide_product_builds():
    # `prod#` is the one primitive of unbounded arity, because a cons cell's
    # payload is a two-field record and nothing else builds one. The pair is
    # what makes `(cons (prod# x xs))` a value rather than a form that only
    # type-checks.
    form = SList(
        [
            Symbol("cons"),
            SList(
                [
                    Symbol("prod#"),
                    Integer(1),
                    SList([Symbol("nil"), SList([Symbol("prod#")])]),
                ]
            ),
        ]
    )

    assert evaluate(form, Environment(BUILTINS)) == listed([Integer(1)])


def test_a_constructed_list_is_the_one_a_quoted_datum_answers_with():
    # The property that matters, as it did for `@bool`: a list has one
    # representation however it was reached, so a built one and a quoted one
    # are the same value and `eq` says so. Were they two representations, a
    # `(cons (prod# x xs))` pattern would match one and not the other.
    built = SList(
        [
            Symbol("cons"),
            SList(
                [
                    Symbol("prod#"),
                    Integer(1),
                    SList(
                        [
                            Symbol("cons"),
                            SList(
                                [
                                    Symbol("prod#"),
                                    Integer(2),
                                    SList([Symbol("nil"), SList([Symbol("prod#")])]),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )
    quoted = SList([Symbol("quote"), SList([Integer(1), Integer(2)])])

    assert evaluate(built, Environment(BUILTINS)) == evaluate(
        quoted, Environment(BUILTINS)
    )
    assert evaluate(
        SList([Symbol("eq"), built, quoted]), Environment(BUILTINS)
    ) == boolean(True)


def test_a_stuck_field_leaves_the_whole_construction_stuck():
    # A primitive reads its arguments, so `prod#` cannot build a product with a
    # free variable in it — and `cons` never runs, because the payload it would
    # read is the residual term.
    payload = SList(
        [Symbol("prod#"), free("n"), SList([Symbol("nil"), SList([Symbol("prod#")])])]
    )
    form = SList([Symbol("cons"), payload])

    assert evaluate(form, Environment(BUILTINS)) == Residual(
        SList(
            [
                Symbol("cons"),
                SList([Symbol("prod#"), free("n"), listed([])]),
            ]
        )
    )


def test_a_wide_product_pattern_binds_its_fields():
    # The other half of representing a product as a tuple of its fields: a
    # `(prod# a b)` pattern is what takes one apart, and the checker reaches it
    # for any product of two fields. A cons cell's own pattern comes here too,
    # one level down — the `prod#` inside `(cons (prod# x xs))` is this pattern
    # over the cell's payload.
    form = SList(
        [
            Symbol("match"),
            SList([Symbol("prod#"), Integer(1), Integer(2)]),
            SList(
                [
                    SList([Symbol("prod#"), Symbol("a"), Symbol("b")]),
                    SList([Symbol("+"), Symbol("a"), Symbol("b")]),
                ]
            ),
        ]
    )

    assert evaluate(form, Environment(BUILTINS)) == Integer(3)


def test_the_list_constructors_read_their_payloads():
    # Unreachable for a checked program, as the boolean case below is: `nil`'s
    # payload is `(@prod)` and `cons`'s is a two-field record, and the checker
    # admits nothing else.
    for form in [
        SList([Symbol("nil"), Integer(1)]),
        SList([Symbol("cons"), Integer(1)]),
        SList([Symbol("cons"), SList([Symbol("prod#"), Integer(1), Integer(2)])]),
    ]:
        with pytest.raises(EvalError):
            evaluate(form, Environment(BUILTINS))


def test_a_boolean_constructor_reads_its_payload():
    # Read rather than discarded, as every primitive reads what it is handed.
    # Unreachable for a checked program: the payload is `(@prod)`, and only the
    # empty product inhabits it.
    with pytest.raises(EvalError):
        evaluate(SList([Symbol("#t"), Integer(1)]), Environment(BUILTINS))


def test_a_match_with_no_clause_matching_is_an_error():
    # Unreachable for a checked program — the checker refuses a `match` that
    # leaves values unmatched — and an error rather than a value for one that
    # was never checked.
    form = SList(
        [
            Symbol("match"),
            Integer(1),
            SList([Integer(2), Integer(3)]),
        ]
    )

    with pytest.raises(EvalError):
        evaluate(form, Environment({}))


@given(symbol_names, self_evaluating_atoms, self_evaluating_atoms)
def test_letrec_bindings_shadow_the_environment_in_the_body(name, inner, outer):
    form = SList(
        [Symbol("letrec"), SList([SList([Symbol(name), inner])]), Symbol(name)]
    )
    assert evaluate(form, Environment({name: outer})) == inner


@given(symbols, symbols, self_evaluating_atoms)
def test_letrec_prebinds_names_to_the_empty_tuple_then_assigns_in_order(
    first, second, value
):
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
    assert evaluate(forward, Environment({})) == unit()
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


# -- free variables and weak head normal form ------------------------------------


def free(name: str, written: str = "@int"):
    return SList(
        [
            Symbol("forall"),
            SList([Symbol("ascribe#"), Symbol(name), Symbol(written)]),
        ]
    )


@given(symbol_names)
def test_a_free_variable_is_its_own_normal_form(name):
    # Nothing bound it and the form says nothing ever will, so evaluation stops
    # here instead of failing. The one place it does: an unbound symbol is
    # still an error, because nothing about it says it was meant to have no
    # value.
    form = free(name)

    assert evaluate(form, Environment({})) == Residual(form)


def test_a_primitive_that_cannot_read_an_argument_gets_stuck():
    # A primitive reads its arguments, so one it cannot read stops it — and the
    # application comes back as the term it could not reduce.
    form = SList([Symbol("+"), Integer(1), free("n")])

    assert evaluate(form, Environment(BUILTINS)) == Residual(
        SList([Symbol("+"), Integer(1), free("n")])
    )


def test_a_stuck_scrutinee_reduces_no_clause():
    # Weak head normal form: reducing one clause would be reducing under a
    # stuck head, and reducing all of them would run clauses the program never
    # reaches. Every body here is unbound, so any of them would have failed.
    test = SList([Symbol("lt"), free("n"), Integer(0)])
    clauses = [
        SList([SList([Symbol("#t"), SList([Symbol("prod#")])]), Symbol("untaken")]),
        SList([SList([Symbol("#f"), SList([Symbol("prod#")])]), Symbol("also")]),
    ]
    form = SList([Symbol("match"), test, *clauses])

    assert evaluate(form, Environment(BUILTINS)) == Residual(
        SList([Symbol("match"), test, *clauses])
    )


def test_a_closure_runs_until_it_needs_the_value():
    # Binding reads nothing, so a residual argument is bound like any other
    # value and the body runs on. What comes back is the body's own stuck term
    # and not the call.
    procedure = SList(
        [
            Symbol("lambda"),
            SList([Symbol("x")]),
            SList([Symbol("+"), Symbol("x"), Symbol("x")]),
        ]
    )
    call = SList([procedure, free("n")])

    assert evaluate(call, Environment(BUILTINS)) == Residual(
        SList([Symbol("+"), free("n"), free("n")])
    )


def test_a_stuck_operator_makes_the_application_stuck():
    call = SList([free("f", "@int"), Integer(1)])

    assert evaluate(call, Environment(BUILTINS)) == Residual(
        SList([free("f", "@int"), Integer(1)])
    )
