from collections.abc import Callable
from itertools import count

import pytest

from Expander import ParseError, expand, parse, parse_type, strip
from Reader import read
from Resolve import Ambiguous, BindingTable, bindings
from Syntax import Char, Expr, Integer, Keyword, Nil, Scope, SList, Symbol
from Syntax.Type import (
    BoolType,
    CharType,
    FunType,
    IntType,
    ListType,
    MuType,
    ProdType,
    SumType,
    Type,
    VariadicStarType,
    VariadicType,
    VarType,
    unfold,
)

# Fixtures inject deterministic counters from the 1300s so no expectation
# depends on the global fresh-uid counter and no uid collides with another
# test file's hand-written scopes.


def counter(start: int) -> Callable[[], Scope]:
    uids = count(start)
    return lambda: Scope(next(uids))


def pipeline(source: str, expand_at: int) -> Expr:
    expanded = expand(read(source), fresh=counter(expand_at))
    return parse(expanded, bindings(expanded))


# -- variables and binding identity --------------------------------------------


def test_lambda_binder_and_reference_share_a_symbol():
    b = Symbol("x", {Scope(1300)})

    assert pipeline("(lambda (x) x)", 1300) == SList([Symbol("lambda"), SList([b]), b])


def test_reference_collapses_to_the_innermost_binder():
    outer = Symbol("x", {Scope(1303)})
    inner = Symbol("x", {Scope(1303), Scope(1304)})

    assert pipeline("(lambda (x) (lambda (x) x))", 1303) == SList(
        [
            Symbol("lambda"),
            SList([outer]),
            SList([Symbol("lambda"), SList([inner]), inner]),
        ]
    )


def test_hygiene_survives_into_the_parsed_tree():
    # The macro introduces a binder spelled like the use-site argument; the
    # substituted reference must come out as the user's binder symbol, with
    # the introduction scope keeping the two spellings apart.
    parsed = pipeline(
        "(lambda (x)"
        " (letrec-syntax ((m (syntax-rules () ((_ e) (lambda (x) e)))))"
        "  (m x)))",
        1306,
    )

    match parsed:
        case SList(
            [
                _,
                SList([Symbol() as outer]),
                SList([_, SList([Symbol() as inner]), Symbol() as body]),
            ]
        ):
            assert body == outer
            assert body != inner
        case _:
            pytest.fail(f"unexpected shape: {parsed}")


def test_letrec_references_collapse_to_their_binders():
    # The occurrence of g inside f's initializer carries f's lambda scope on
    # top of the letrec region pre-parse; parsing discharges it to exactly
    # the binder's own symbol.
    f = Symbol("f", {Scope(1310)})
    g = Symbol("g", {Scope(1310)})
    a = Symbol("a", {Scope(1310), Scope(1311)})
    b = Symbol("b", {Scope(1310), Scope(1312)})

    assert pipeline(
        "(letrec ((f (lambda (a) (g a))) (g (lambda (b) b))) (f z))", 1310
    ) == SList(
        [
            Symbol("letrec"),
            SList(
                [
                    SList([f, SList([Symbol("lambda"), SList([a]), SList([g, a])])]),
                    SList([g, SList([Symbol("lambda"), SList([b]), b])]),
                ]
            ),
            SList([f, Symbol("z")]),
        ]
    )


def test_unbound_references_parse_to_bare_spellings():
    x = Symbol("x", {Scope(1315)})

    assert pipeline("(lambda (x) (le x y))", 1315) == SList(
        [
            Symbol("lambda"),
            SList([x]),
            SList([Symbol("le"), x, Symbol("y")]),
        ]
    )


# -- quote ---------------------------------------------------------------------


def test_quote_yields_a_scope_free_datum():
    x = Symbol("x", {Scope(1320)})

    assert pipeline("(lambda (x) '(x 5))", 1320) == SList(
        [
            Symbol("lambda"),
            SList([x]),
            SList([Symbol("quote"), SList([Symbol("x"), Integer(5)])]),
        ]
    )


def test_strip_discharges_scopes_everywhere():
    scoped = SList([Symbol("x", {Scope(1324)}), SList([Symbol("y", {Scope(1325)})])])

    assert strip(scoped) == SList([Symbol("x"), SList([Symbol("y")])])


# -- inert core forms ------------------------------------------------------------


def test_if_parses_positionally_with_a_bare_head():
    x = Symbol("x", {Scope(1330)})

    assert pipeline("(lambda (x) (if x x x))", 1330) == SList(
        [
            Symbol("lambda"),
            SList([x]),
            SList([Symbol("if"), x, x, x]),
        ]
    )


def test_choice_parses_as_an_application_with_a_free_head():
    x = Symbol("x", {Scope(1335)})

    assert pipeline("(lambda (x) (choice 1 2 5 x))", 1335) == SList(
        [
            Symbol("lambda"),
            SList([x]),
            SList([Symbol("choice"), Integer(1), Integer(2), Integer(5), x]),
        ]
    )


def test_constants_are_leaves():
    assert pipeline('(f 5 #t prod# "hi")', 1340) == SList(
        [
            Symbol("f"),
            Integer(5),
            Symbol("#t"),
            Symbol("prod#"),
            SList([Char(ord("h")), Char(ord("i"))]),
        ]
    )


# -- refusals --------------------------------------------------------------------


def test_empty_combination_is_refused():
    with pytest.raises(ParseError):
        parse(SList([]), BindingTable())


def test_unexpanded_macro_forms_are_refused():
    with pytest.raises(ParseError):
        parse(
            read("(letrec-syntax ((m (syntax-rules () ((_) x)))) y)"),
            BindingTable(),
        )


@pytest.mark.parametrize(
    "source",
    [
        "(if x y)",
        "(lambda x x)",
        "(quote)",
        "(letrec ((f)) x)",
        "(ascribe# x @int)",
        "(define x)",
    ],
)
def test_malformed_core_shapes_are_refused(source: str):
    with pytest.raises(ParseError):
        parse(read(source), BindingTable())


def test_binder_missing_from_the_store_is_refused():
    expanded = expand(read("(lambda (x) x)"), fresh=counter(1345))

    with pytest.raises(ParseError):
        parse(expanded, BindingTable())


def test_ambiguous_references_propagate():
    table = BindingTable(
        [
            Symbol("x", {Scope(1350), Scope(1351)}),
            Symbol("x", {Scope(1350), Scope(1352)}),
        ]
    )

    with pytest.raises(Ambiguous):
        parse(Symbol("x", {Scope(1350), Scope(1351), Scope(1352)}), table)


# -- end to end ------------------------------------------------------------------


def test_macro_expanded_programs_parse():
    # The same program the stochastic analysis consumes: the macro-introduced
    # identity and both diverging lambdas parse, every residual reference is
    # discharged, and the body's head lands on the user's binder.
    expanded = expand(
        read(
            "(letrec-syntax ((id (syntax-rules () ((_) (lambda (v) v)))))"
            " ((lambda (f) (f y))"
            "  (choice 1 3 (id) ((lambda (d) (d d)) (lambda (d) (d d))))))"
        ),
        fresh=counter(1360),
    )
    parsed = parse(expanded, bindings(expanded))

    match parsed:
        case SList(
            [
                SList(
                    [
                        _,
                        SList([Symbol() as f]),
                        SList([Symbol() as head, Symbol() as free]),
                    ]
                ),
                _,
            ]
        ):
            assert head == f
            assert free == Symbol("y")
        case _:
            pytest.fail(f"unexpected shape: {parsed}")


# -- type ascription -----------------------------------------------------------


def test_an_ascribed_lambda_parameter_keeps_its_wrapper():
    # The ascription survives into the parsed tree: it is what the type
    # checker reads the binder's type off of. The keyword and the type operand
    # are both normalized to bare symbols, exactly as `lambda` itself is.
    b = Symbol("x", {Scope(1360)})

    assert pipeline("(lambda ((ascribe# x @int)) x)", 1360) == SList(
        [
            Symbol("lambda"),
            SList([SList([Symbol("ascribe#"), b, Symbol("@int")])]),
            b,
        ]
    )


def test_an_ascribed_letrec_binder_keeps_its_wrapper():
    b = Symbol("n", {Scope(1361)})

    assert pipeline("(letrec (((ascribe# n @int) 5)) n)", 1361) == SList(
        [
            Symbol("letrec"),
            SList(
                [SList([SList([Symbol("ascribe#"), b, Symbol("@int")]), Integer(5)])]
            ),
            b,
        ]
    )


def test_an_ascribed_binder_and_its_reference_share_a_symbol():
    # The whole point of resolving the wrapped name: the body's occurrence
    # collapses to the very binder the ascription wraps.
    parsed = pipeline("(lambda ((ascribe# x @int)) x)", 1362)

    match parsed:
        case SList([_, SList([SList([_, bound, _])]), reference]):
            assert bound is not None and bound == reference
        case _:
            pytest.fail(f"unexpected parse: {parsed}")


def test_an_ascribed_type_operand_is_not_resolved():
    # `@int` is registered nowhere, so were it read as a reference `parse`
    # would either refuse it or rewrite it. It is stripped instead, and comes
    # through as the bare spelling it was written as — `Symbol` equality
    # covers scopes, so this also pins that it carries none.
    parsed = pipeline("(lambda ((ascribe# x @int)) x)", 1363)

    match parsed:
        case SList([_, SList([SList([_, _, ty])]), _]):
            assert ty == Symbol("@int")
        case _:
            pytest.fail(f"unexpected parse: {parsed}")


def test_a_malformed_ascribed_binder_is_refused():
    # `parse` guards the shape itself rather than trusting what reached it:
    # a wrapper whose type operand is a list is structure out of place. The
    # tree is hand-built because `expand` refuses this one first.
    malformed = SList(
        [
            Symbol("lambda"),
            SList(
                [
                    SList(
                        [
                            Symbol("ascribe#"),
                            Symbol("x"),
                            SList([Symbol("@list"), Symbol("@int")]),
                        ]
                    )
                ]
            ),
            Symbol("x"),
        ]
    )

    with pytest.raises(ParseError):
        parse(malformed, BindingTable())


def test_a_macro_introduced_ascription_is_recognized():
    # A template that introduces an ascription gives the keyword and the type
    # operand scopes of their own, so every reader has to match the head by
    # spelling rather than by symbol identity. `parse` resolving the wrapped
    # binder is what proves `bindings` found it through the wrapper.
    source = (
        "(letrec-syntax ((m (syntax-rules () ((m t) (lambda ((ascribe# y t)) y)))))"
        " (m @int))"
    )
    expanded = expand(read(source), fresh=counter(1365))

    # The hazard, pinned: in the expanded tree the keyword is not bare.
    match expanded:
        case SList([_, SList([SList([Symbol() as keyword, *_])]), _]):
            assert keyword.identifier == "ascribe#"
            assert keyword.scopes != set()
        case _:
            pytest.fail(f"unexpected expansion: {expanded}")

    match parse(expanded, bindings(expanded)):
        case SList([head, SList([SList([_, bound, ty])]), reference]):
            assert head == Symbol("lambda")
            assert ty == Symbol("@int")
            assert bound == reference
        case other:
            pytest.fail(f"unexpected parse: {other}")


# -- types -----------------------------------------------------------------------


def type_of(source: str) -> Type:
    # A written type read and then parsed, which is how every type reaches
    # `parse_type` from source.
    return parse_type(read(source))


def test_every_ground_type_is_written_as_a_bare_symbol():
    assert type_of("@int") == IntType()
    assert type_of("@char") == CharType()
    assert type_of("@bool") == BoolType()


def test_a_list_type_is_written_over_its_element():
    assert type_of("(@list @int)") == ListType(IntType())
    assert type_of("(@list (@list @char))") == ListType(ListType(CharType()))


def test_the_nullary_product_is_written_with_no_fields():
    assert type_of("(@prod)") == ProdType()


def test_a_positional_product_keeps_its_fields_in_order():
    assert type_of("(@prod @int @bool)") == ProdType([IntType(), BoolType()])


def test_a_named_product_is_a_record_of_bare_names():
    record = type_of("(@prod (head @int) (tail @bool))")

    assert record == ProdType({Symbol("head"): IntType(), Symbol("tail"): BoolType()})
    assert isinstance(record, ProdType)
    assert record.names() == (Symbol("head"), Symbol("tail"))


def test_a_field_written_as_a_type_form_stays_positional():
    # The disambiguation: `(@list @int)` is a type and not a field named
    # `@list`, which is what the `@` spelling tells apart.
    assert type_of("(@prod (@list @int))") == ProdType([ListType(IntType())])


@pytest.mark.parametrize(
    "source",
    ["(@prod @int (tail @bool))", "(@prod (head @int) @bool)"],
)
def test_a_product_mixing_named_and_positional_fields_is_refused(source: str):
    with pytest.raises(ParseError):
        type_of(source)


def test_a_product_repeating_a_field_name_is_refused():
    with pytest.raises(ParseError):
        type_of("(@prod (x @int) (x @bool))")


def test_a_sum_is_written_as_its_constructors():
    assert type_of("(@sum (zero (@prod)) (succ @int))") == SumType(
        {Symbol("zero"): ProdType(), Symbol("succ"): IntType()}
    )
    assert type_of("(@sum (only @int))") == SumType({Symbol("only"): IntType()})


@pytest.mark.parametrize("source", ["(@sum)", "(@sum @int)", "(@sum (@list @int))"])
def test_a_sum_is_refused_without_a_constructor_for_every_payload(source: str):
    # The empty sum is refused here rather than by `SumType`, so the diagnosis
    # is this phase's; a payload written without a name has no constructor to
    # be reached by.
    with pytest.raises(ParseError):
        type_of(source)


def test_a_sum_repeating_a_constructor_is_refused():
    with pytest.raises(ParseError):
        type_of("(@sum (c @int) (c @bool))")


def test_the_variadic_runs_are_written_over_one_type():
    assert type_of("(@variadic @int)") == VariadicType(IntType())
    assert type_of("(@variadic* @int)") == VariadicStarType(IntType())


def test_a_function_type_is_written_run_then_return():
    assert type_of("(@fun (@prod @int @int) @bool)") == FunType(
        return_type=BoolType(), argument_type=ProdType([IntType(), IntType()])
    )
    assert type_of("(@fun (@prod) @bool)") == FunType(return_type=BoolType())
    assert type_of("(@fun (@prod (n @int)) @int)") == FunType(
        return_type=IntType(), argument_type=ProdType({Symbol("n"): IntType()})
    )


def test_a_function_type_is_written_over_a_variadic_run():
    assert type_of("(@fun (@variadic @int) @int)") == FunType(
        return_type=IntType(), argument_type=VariadicType(IntType())
    )
    assert type_of("(@fun (@variadic* @int) @int)") == FunType(
        return_type=IntType(), argument_type=VariadicStarType(IntType())
    )


def test_a_function_type_over_a_type_that_is_no_run_is_refused():
    # `@int` classifies values, and a function is applied to an argument list.
    with pytest.raises(ParseError):
        type_of("(@fun @int @int)")


@pytest.mark.parametrize(
    "source",
    ["(@fun)", "(@fun @int)", "(@fun (@prod) @int @int)"],
)
def test_a_function_type_takes_exactly_a_run_and_a_return(source: str):
    with pytest.raises(ParseError):
        type_of(source)


def test_a_recursive_type_binds_its_variable_in_its_body():
    # The `nat()` fixture of `test_recursive_types.py`, written out.
    assert type_of("(@mu a (@sum (zero (@prod)) (succ a)))") == MuType(
        Symbol("a"),
        SumType({Symbol("zero"): ProdType(), Symbol("succ"): VarType(Symbol("a"))}),
    )


def test_a_recursive_record_is_written_through_its_payloads():
    # The `ints()` fixture, whose recursive position sits two sorts down.
    source = "(@mu a (@sum (nil (@prod)) (cons (@prod (head @int) (tail a)))))"

    assert type_of(source) == MuType(
        Symbol("a"),
        SumType(
            {
                Symbol("nil"): ProdType(),
                Symbol("cons"): ProdType(
                    {
                        Symbol("head"): IntType(),
                        Symbol("tail"): VarType(Symbol("a")),
                    }
                ),
            }
        ),
    )


def test_a_parsed_recursive_type_unfolds():
    # What says the symbols the parser emits are the ones substitution needs:
    # `Symbol` equality covers scopes, so a binder and its occurrences have to
    # normalize to one value for `unfold` to replace anything at all.
    parsed = type_of("(@mu a (@list a))")

    assert isinstance(parsed, MuType)
    assert unfold(parsed) == ListType(parsed)


def test_a_nested_recursive_type_binds_the_same_spelling_again():
    inner = MuType(Symbol("a"), ProdType([VarType(Symbol("a"))]))

    assert type_of("(@mu a (@prod (@mu a (@prod a))))") == MuType(
        Symbol("a"), ProdType([inner])
    )


def test_an_outer_binding_reaches_inside_an_inner_one():
    # The spellings accumulate, so a variable the outer `@mu` binds is still
    # one inside the inner's body.
    assert type_of("(@mu a (@prod (@mu b (@prod a))))") == MuType(
        Symbol("a"),
        ProdType([MuType(Symbol("b"), ProdType([VarType(Symbol("a"))]))]),
    )


def test_a_bound_type_variable_is_read_where_it_is_passed():
    # The parameter the `@mu` arm threads, read on its own.
    assert parse_type(read("a"), frozenset({"a"})) == VarType(Symbol("a"))


@pytest.mark.parametrize("source", ["a", "(@list a)", "(@prod (x a))"])
def test_a_free_type_variable_is_refused(source: str):
    # Nothing binds the spelling, so it stands for no type. Refused here rather
    # than built into a `VarType` every later reading would refuse instead.
    with pytest.raises(ParseError):
        type_of(source)


def test_a_recursive_type_bound_to_a_constructor_spelling_is_refused():
    with pytest.raises(ParseError):
        type_of("(@mu @int (@list @int))")


def test_a_type_carrying_scopes_parses_as_the_bare_one():
    # A macro-introduced type carries the scopes of wherever it was written, so
    # every reading is by spelling -- as `ascription` reads its own head.
    scoped = read("(@mu a (@sum (zero (@prod)) (succ a)))")
    scoped.add(Scope(1370))

    assert parse_type(scoped) == type_of("(@mu a (@sum (zero (@prod)) (succ a)))")


@pytest.mark.parametrize(
    "source",
    ["@list", "@prod", "@sum", "@fun", "@mu", "@variadic", "@foo"],
)
def test_a_type_constructor_spelling_is_not_a_type_on_its_own(source: str):
    with pytest.raises(ParseError):
        type_of(source)


@pytest.mark.parametrize(
    "source",
    [
        "()",
        "(@foo @int)",
        "(@list)",
        "(@list @int @int)",
        "(@variadic)",
        "(@variadic* @int @int)",
        "(@mu a)",
        "(@mu (a) @int)",
        "((@list @int) @int)",
    ],
)
def test_a_malformed_or_unknown_type_form_is_refused(source: str):
    with pytest.raises(ParseError):
        type_of(source)


def test_a_datum_is_not_a_type():
    for datum in [Integer(1), Char(97), Keyword("x"), Nil(), SList([])]:
        with pytest.raises(ParseError):
            parse_type(datum)
