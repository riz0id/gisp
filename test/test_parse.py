from collections.abc import Callable
from itertools import count

import pytest

from Expander import ParseError, expand, parse, strip, type_syntax
from Reader import read
from Resolve import Ambiguous, BindingTable, bindings
from Syntax import Char, Expr, Integer, Keyword, Scope, SList, Symbol

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


def test_a_match_parses_positionally_with_a_bare_head():
    # The clause scopes are discharged by resolution: all three occurrences of
    # `x` resolve to the one parameter, whatever scopes the clause gave them.
    x = Symbol("x", {Scope(1330)})

    assert pipeline(
        "(lambda (x) (match x ((#t (prod#)) x) ((#f (prod#)) x)))", 1330
    ) == SList(
        [
            Symbol("lambda"),
            SList([x]),
            SList(
                [
                    Symbol("match"),
                    x,
                    SList([SList([Symbol("#t"), SList([Symbol("prod#")])]), x]),
                    SList([SList([Symbol("#f"), SList([Symbol("prod#")])]), x]),
                ]
            ),
        ]
    )


def test_a_pattern_resolves_its_binders_and_keeps_its_constructor():
    # A constructor head resolves as a reference and comes back as itself when
    # nothing binds it, which is what makes `cons` and the `prod#` its payload
    # pattern is written with readable as the constructors they name.
    assert pipeline("(match xs ((cons (prod# x r)) x))", 1332) == SList(
        [
            Symbol("match"),
            Symbol("xs"),
            SList(
                [
                    SList(
                        [
                            Symbol("cons"),
                            SList(
                                [
                                    Symbol("prod#"),
                                    Symbol("x", {Scope(1332)}),
                                    Symbol("r", {Scope(1332)}),
                                ]
                            ),
                        ]
                    ),
                    Symbol("x", {Scope(1332)}),
                ]
            ),
        ]
    )


def test_a_literal_pattern_is_a_leaf():
    assert pipeline("(match n (0 1))", 1334) == SList(
        [
            Symbol("match"),
            Symbol("n"),
            SList([Integer(0), Integer(1)]),
        ]
    )


def test_a_pattern_head_that_is_not_a_symbol_is_refused():
    # Expansion leaves it alone — a pattern's head is re-emitted as written,
    # because resolution is what decides whether it names a constructor — so
    # this is the one malformed pattern the parse phase is what refuses. Every
    # other one is caught before this phase sees it.
    expanded = expand(read("(match n (((ascribe# x @int)) b))"), fresh=counter(1336))

    with pytest.raises(ParseError):
        parse(expanded, bindings(expanded))


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
        "(match x)",
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


def test_a_macro_introduced_ascription_takes_a_compound_type():
    # The use site supplies a whole type, spliced beneath a template `ascribe#`:
    # the operand carries use-site scopes and the keyword template ones. The
    # type's own binders are scoped where the type was written, so the
    # quantified variable still resolves to the quantifier that bound it.
    source = (
        "(letrec-syntax ((m (syntax-rules () ((m t) (lambda ((ascribe# y t)) y)))))"
        " (m (forall# (a) (@fun (@prod a) a))))"
    )
    expanded = expand(read(source), fresh=counter(1375))

    match parse(expanded, bindings(expanded)):
        case SList([_, SList([SList([_, _, ty])]), _]):
            pass
        case other:
            pytest.fail(f"unexpected parse: {other}")

    match ty:
        case SList([_, SList([bound]), SList([_, SList([_, argument]), returned])]):
            pass
        case other:
            pytest.fail(f"not a parsed forall#: {other}")

    assert bound.identifier == "a"
    assert bound == argument == returned


def test_a_macro_introduced_malformed_type_is_refused():
    # The template is well formed and the use site is not, so the refusal has to
    # come from the type the use site supplied.
    source = (
        "(letrec-syntax ((m (syntax-rules () ((m t) (lambda ((ascribe# y t)) y)))))"
        " (m (@list @int @int)))"
    )
    expanded = expand(read(source), fresh=counter(1380))

    with pytest.raises(ParseError):
        parse(expanded, bindings(expanded))


def test_the_binding_table_round_trips_through_a_parsed_type():
    # `bindings` is called on an expanded tree to parse it, and on the parsed
    # tree by every later reader, so the two tables have to agree. A type binder
    # reaches the second only if `bindings` reads `@mu` and `forall#` as the
    # binding forms they are — and the parse phase resolved against the first,
    # so a binder missing from the second is one no later reader could find.
    source = (
        "(lambda ((ascribe# f (forall# (a) (@fun (@prod (@mu b (@list b))) a)))) f)"
    )
    expanded = expand(read(source), fresh=counter(1385))
    before = bindings(expanded)
    parsed = parse(expanded, before)

    assert set(bindings(parsed)) == set(before)
    assert {binder.identifier for binder in before} == {"f", "a", "b"}


# -- types: written form and binding ---------------------------------------------
#
# The parse phase checks how a type is written and resolves the variables within
# it. What a type denotes — its rank, its kinds, its guardedness — is read by
# `TypeCheck.Elaborate`, and `test_elaborate.py` covers that.


def type_operand(source: str, expand_at: int = 1600) -> Expr:
    # The parsed type operand of `(lambda ((ascribe# x <source>)) x)`, which is
    # how a written type reaches the parse phase: through a binder position, in
    # a real program, after expansion has scoped it.
    program = f"(lambda ((ascribe# x {source})) x)"
    expanded = expand(read(program), fresh=counter(expand_at))

    match parse(expanded, bindings(expanded)):
        case SList([_, SList([SList([_, _, ty])]), _]):
            return ty
        case parsed:
            raise AssertionError(f"not an ascribed lambda: {parsed}")


def test_a_ground_type_is_emitted_with_an_empty_scope_set():
    # A type constructor's identifier names no variable and resolves against
    # nothing, so it is re-emitted bare exactly as `parse` rebuilds `lambda`.
    for source in ["@int", "@char", "@bool"]:
        assert type_operand(source) == Symbol(source)


def test_a_compound_type_keeps_its_shape_and_empties_its_scopes():
    assert type_operand("(@list @int)") == SList([Symbol("@list"), Symbol("@int")])
    assert type_operand("(@fun (@prod @int) @bool)") == SList(
        [
            Symbol("@fun"),
            SList([Symbol("@prod"), Symbol("@int")]),
            Symbol("@bool"),
        ]
    )


def test_a_field_name_is_emitted_with_an_empty_scope_set():
    # A record's field name labels a position; it binds nothing and refers to
    # nothing, so it is emitted bare like a constructor's identifier.
    assert type_operand("(@prod (head @int) (tail @bool))") == SList(
        [
            Symbol("@prod"),
            SList([Symbol("head"), Symbol("@int")]),
            SList([Symbol("tail"), Symbol("@bool")]),
        ]
    )


def test_a_type_variable_resolves_to_its_binder():
    # The whole point of the merge: a type variable goes through the one binding
    # mechanism, so its occurrence comes back as the very binder `@mu` minted.
    match type_operand("(@mu a (@list a))"):
        case SList([_, bound, SList([_, occurrence])]):
            pass
        case parsed:
            raise AssertionError(f"not a parsed @mu: {parsed}")

    assert bound.identifier == "a"
    assert bound == occurrence
    assert bound.scopes != set()


def test_a_quantified_variable_resolves_to_its_binder():
    match type_operand("(forall# (a) (@fun (@prod a) a))"):
        case SList([_, SList([bound]), SList([_, SList([_, argument]), returned])]):
            pass
        case parsed:
            raise AssertionError(f"not a parsed forall#: {parsed}")

    assert bound.identifier == "a"
    assert bound == argument == returned


def test_nested_recursive_types_resolve_to_their_own_binders():
    # An inner `@mu` reusing an identifier is a distinct binder, so the inner
    # body's occurrence is the inner binder and not the outer one. This is what
    # the set of identifier strings could only approximate.
    match type_operand("(@mu a (@prod (@mu a (@prod a))))"):
        case SList([_, outer, SList([_, SList([_, inner, SList([_, occurrence])])])]):
            pass
        case parsed:
            raise AssertionError(f"not a nested @mu: {parsed}")

    assert outer.identifier == inner.identifier == "a"
    assert outer != inner
    assert occurrence == inner


def test_an_outer_type_binder_reaches_inside_an_inner_one():
    match type_operand("(@mu a (@prod (@mu b (@prod a))))"):
        case SList([_, outer, SList([_, SList([_, _, SList([_, occurrence])])])]):
            pass
        case parsed:
            raise AssertionError(f"not a nested @mu: {parsed}")

    assert occurrence == outer


@pytest.mark.parametrize("source", ["a", "(@list a)", "(@prod (x a))"])
def test_a_free_type_variable_is_refused(source: str):
    # Nothing binds the identifier, so it stands for no type. Refused here
    # rather than emitted for a later reading to refuse instead.
    with pytest.raises(ParseError):
        type_operand(source)


@pytest.mark.parametrize(
    "source",
    ["(@prod @int (tail @bool))", "(@prod (head @int) @bool)"],
)
def test_a_product_mixing_named_and_positional_fields_is_refused(source: str):
    with pytest.raises(ParseError):
        type_operand(source)


def test_a_product_repeating_a_field_name_is_refused():
    with pytest.raises(ParseError):
        type_operand("(@prod (x @int) (x @bool))")


@pytest.mark.parametrize("source", ["(@sum)", "(@sum @int)", "(@sum (@list @int))"])
def test_a_sum_is_refused_without_a_constructor_for_every_payload(source: str):
    # The empty sum is refused here rather than by `SumType`, so the diagnosis
    # is this phase's; a payload written without a name has no constructor to
    # be reached by.
    with pytest.raises(ParseError):
        type_operand(source)


def test_a_sum_repeating_a_constructor_is_refused():
    with pytest.raises(ParseError):
        type_operand("(@sum (c @int) (c @bool))")


@pytest.mark.parametrize(
    "source",
    ["(@fun)", "(@fun @int)", "(@fun (@prod) @int @int)"],
)
def test_a_function_type_takes_exactly_a_run_and_a_return(source: str):
    with pytest.raises(ParseError):
        type_operand(source)


@pytest.mark.parametrize(
    "source",
    ["@list", "@prod", "@sum", "@fun", "@mu", "@variadic", "@foo"],
)
def test_a_type_constructor_identifier_is_not_a_type_on_its_own(source: str):
    with pytest.raises(ParseError):
        type_operand(source)


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
        "((@list @int) @int)",
        "(forall#)",
        "(forall# (a))",
        "(forall# a @int)",
        "(forall# (a) @int @int)",
        "(forall# (a) b)",
        "(forall# (a) (@list))",
    ],
)
def test_a_malformed_or_unknown_type_form_is_refused(source: str):
    with pytest.raises(ParseError):
        type_operand(source)


def test_a_datum_is_not_a_type():
    for datum in [Integer(1), Char(97), Keyword("x"), SList([])]:
        with pytest.raises(ParseError):
            type_syntax(datum, BindingTable())


def test_a_nested_quantifier_parses_and_is_left_to_elaboration():
    # Rank-1 is a rule about what a type means, so the parse phase has no
    # opinion on it: a `forall#` written inside a type resolves and is emitted,
    # and `test_elaborate.py` covers its refusal.
    match type_operand("(@list (forall# (a) a))"):
        case SList([head, SList([quantifier, SList([bound]), occurrence])]):
            pass
        case parsed:
            raise AssertionError(f"not a nested forall#: {parsed}")

    assert head == Symbol("@list")
    assert quantifier == Symbol("forall#")
    assert bound == occurrence


# -- free variables --------------------------------------------------------------


def test_forall_resolves_its_binder_and_reads_its_type():
    # Read exactly as a lambda parameter's position is: the binder resolves and
    # the type operand is read as a type, which is what drops the scope the
    # expansion gave the operand.
    assert pipeline("(forall (ascribe# n @int))", 1380) == SList(
        [
            Symbol("forall"),
            SList([Symbol("ascribe#"), Symbol("n"), Symbol("@int")]),
        ]
    )


def test_a_bare_reference_reaches_a_free_variable():
    # The binder is in the table, so an ordinary reference resolves to it
    # rather than falling back to its own spelling.
    free = Symbol("n")

    assert pipeline("(lt (forall (ascribe# n @int)) n)", 1382) == SList(
        [
            Symbol("lt"),
            SList(
                [Symbol("forall"), SList([Symbol("ascribe#"), free, Symbol("@int")])]
            ),
            free,
        ]
    )


def test_a_malformed_forall_is_refused_by_the_parse_phase():
    # The expander refuses one first, so this is reached only by syntax built
    # directly. The arm is what keeps a `forall` from falling into the
    # application arm below it and being read as a call to an unbound name.
    for form in [
        SList([Symbol("forall")]),
        SList([Symbol("forall"), Symbol("n"), Integer(1)]),
    ]:
        with pytest.raises(ParseError):
            parse(form, BindingTable())
