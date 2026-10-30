from collections.abc import Callable
from itertools import count

import pytest

from Expander import KEYWORDS, ExpandError, expand
from Reader import read
from Resolve import Unbound, bindings, resolve
from Syntax import Integer, Scope, SList, Symbol

# Fixtures in this file never depend on the global fresh-uid counter: every
# deterministic expectation injects its own counter, drawn from the 900s so
# the uids cannot collide with any other test file's hand-written scopes.


def counter(start: int) -> Callable[[], Scope]:
    uids = count(start)
    return lambda: Scope(next(uids))


def test_lambda_reproduces_the_canonical_shadowing_fixture():
    # (lambda (x) (lambda (x) x)) expands to the exact shape every resolver
    # test writes by hand: outer binder {901}, inner binder and occurrence
    # {901, 902}. The outer keyword sits in the empty top-level region and
    # stays scopeless; the inner keyword sits in the outer body region and
    # carries {901} but never its own form's scope.
    source = read("(lambda (x) (lambda (x) x))")
    expanded = expand(source, fresh=counter(901))

    expected = SList(
        [
            Symbol("lambda"),
            SList([Symbol("x", {Scope(901)})]),
            SList(
                [
                    Symbol("lambda", {Scope(901)}),
                    SList([Symbol("x", {Scope(901), Scope(902)})]),
                    Symbol("x", {Scope(901), Scope(902)}),
                ]
            ),
        ]
    )

    assert expanded == expected

    table = bindings(expanded)
    occurrence = Symbol("x", {Scope(901), Scope(902)})
    assert resolve(occurrence, table).binder == Symbol("x", {Scope(901), Scope(902)})
    assert resolve(occurrence, table).binder != Symbol("x", {Scope(901)})


def test_multi_parameter_lambda_shares_one_scope():
    # One binding form, one scope: both parameters carry {905}, and each
    # body occurrence still resolves to its own binder because resolution
    # filters by spelling before it compares scope sets.
    expanded = expand(read("(lambda (a b) (a b))"), fresh=counter(905))
    table = bindings(expanded)

    a_binder = Symbol("a", {Scope(905)})
    b_binder = Symbol("b", {Scope(905)})
    assert set(table) == {a_binder, b_binder}
    assert resolve(Symbol("a", {Scope(905)}), table).binder == a_binder
    assert resolve(Symbol("b", {Scope(905)}), table).binder == b_binder


def test_letrec_bindings_are_mutually_recursive():
    # The letrec scope (910) covers both right-hand sides and the body, so
    # each initializer sees the other binding and the body sees both.
    expanded = expand(
        read("(letrec ((f (lambda (x) (g x))) (g (lambda (y) (f y)))) (f z))"),
        fresh=counter(910),
    )
    table = bindings(expanded)

    f_binder = Symbol("f", {Scope(910)})
    g_binder = Symbol("g", {Scope(910)})
    assert f_binder in table
    assert g_binder in table

    # g inside f's initializer carries the letrec scope plus f's lambda
    # scope; f inside g's initializer likewise with g's lambda scope.
    assert resolve(Symbol("g", {Scope(910), Scope(911)}), table).binder == g_binder
    assert resolve(Symbol("f", {Scope(910), Scope(912)}), table).binder == f_binder
    assert resolve(Symbol("f", {Scope(910)}), table).binder == f_binder

    with pytest.raises(Unbound):
        resolve(Symbol("z", {Scope(910)}), table)


def test_a_lambda_parameter_shadows_a_letrec_binder():
    expanded = expand(
        read("(letrec ((f (lambda (x) x))) (lambda (f) f))"), fresh=counter(915)
    )
    table = bindings(expanded)

    letrec_binder = Symbol("f", {Scope(915)})
    parameter = Symbol("f", {Scope(915), Scope(917)})
    assert resolve(parameter, table).binder == parameter
    assert resolve(parameter, table).binder != letrec_binder
    assert resolve(Symbol("f", {Scope(915)}), table).binder == letrec_binder


def test_a_boolean_match_scopes_each_clause_separately():
    # The shape an `if` used to be written as, now written as what it is.
    #
    # Each clause is a region, so each body carries the clause's own scope
    # beside the lambda's. A pattern is not a region: its constructor head is
    # an identifier of the enclosing one and carries only the lambda's scope,
    # which is what lets resolution decide whether the spelling names a
    # constructor or a binding the program made.
    expanded = expand(
        read("(lambda (x) (match x ((#t) x) ((#f) x)))"), fresh=counter(920)
    )

    parameter = Symbol("x", {Scope(920)})

    assert expanded == SList(
        [
            Symbol("lambda"),
            SList([parameter]),
            SList(
                [
                    Symbol("match", {Scope(920)}),
                    Symbol("x", {Scope(920)}),
                    SList(
                        [
                            SList([Symbol("#t", {Scope(920)})]),
                            Symbol("x", {Scope(920), Scope(921)}),
                        ]
                    ),
                    SList(
                        [
                            SList([Symbol("#f", {Scope(920)})]),
                            Symbol("x", {Scope(920), Scope(922)}),
                        ]
                    ),
                ]
            ),
        ]
    )


def test_if_is_not_a_form_of_the_language():
    # `match` is the only branching form, so `if` names nothing: the spelling
    # is an ordinary identifier, bindable like any other, and a form headed by
    # it is an application.
    assert "if" not in KEYWORDS

    expanded = expand(read("(lambda (if) (if 1 2 3))"), fresh=counter(926))
    bound = Symbol("if", {Scope(926)})

    assert expanded == SList(
        [
            Symbol("lambda"),
            SList([bound]),
            SList([bound, Integer(1), Integer(2), Integer(3)]),
        ]
    )


def test_each_match_clause_is_a_region_of_its_own():
    # The one hygiene shape no other form has. `lambda` and `letrec` mint one
    # scope for the whole form because their binders share one region; two
    # clauses are two regions, so two clauses binding the one spelling are two
    # distinct binders.
    expanded = expand(read("(match xs ((cons x r) x) ((nil) x))"), fresh=counter(930))

    assert expanded == SList(
        [
            Symbol("match"),
            Symbol("xs"),
            SList(
                [
                    SList(
                        [
                            Symbol("cons"),
                            Symbol("x", {Scope(930)}),
                            Symbol("r", {Scope(930)}),
                        ]
                    ),
                    Symbol("x", {Scope(930)}),
                ]
            ),
            SList([SList([Symbol("nil")]), Symbol("x", {Scope(931)})]),
        ]
    )


def test_a_pattern_binder_takes_an_ascription():
    # A pattern binder is scoped exactly as a lambda parameter is, type operand
    # and all: the clause mints 935 and the type operand its own 936.
    expanded = expand(read("(match n ((ascribe# m @int) m))"), fresh=counter(935))

    assert expanded == SList(
        [
            Symbol("match"),
            Symbol("n"),
            SList(
                [
                    SList(
                        [
                            Symbol("ascribe#"),
                            Symbol("m", {Scope(935)}),
                            Symbol("@int", {Scope(936)}),
                        ]
                    ),
                    Symbol("m", {Scope(935)}),
                ]
            ),
        ]
    )


def test_quote_passes_through_untouched():
    # Quoted forms are data: no scopes are fabricated inside them, even when
    # the quote sits inside a binding region.
    assert expand(read("'x")) == SList([Symbol("quote"), Symbol("x")])

    expanded = expand(read("(lambda (x) 'x)"), fresh=counter(925))
    assert expanded == SList(
        [
            Symbol("lambda"),
            SList([Symbol("x", {Scope(925)})]),
            SList([Symbol("quote"), Symbol("x")]),
        ]
    )


@pytest.mark.parametrize(
    "source",
    [
        "(lambda (x x) x)",
        "(letrec ((f f) (f f)) f)",
        "(lambda (match) match)",
        "(letrec ((lambda (lambda (x) x))) x)",
        "(lambda x x)",
        "(lambda (x))",
        "(letrec ((f)) f)",
        "(letrec (f) f)",
        "(lambda ((ascribe# x)) x)",
        "(letrec (((ascribe# n) 5)) n)",
        "(lambda ((ascribe# x @int @int)) x)",
        "(lambda (((ascribe# x) @int)) x)",
        "(ascribe# x @int)",
        "(lambda (ascribe#) ascribe#)",
        "(letrec ((ascribe# 5)) ascribe#)",
        "(define)",
        "(define x)",
        "(define x 1 2)",
        "(lambda (define) define)",
        "(letrec ((define 5)) define)",
        "(match)",
        "(match n)",
        "(match n (p))",
        "(match n (p b c))",
        "(match n (() b))",
        "(match n ((cons x x) b))",
        "(lambda (match) match)",
        "(letrec ((match 5)) match)",
    ],
)
def test_malformed_core_forms_are_refused(source: str):
    with pytest.raises(ExpandError):
        expand(read(source))


def test_an_ascribed_binder_takes_a_compound_type():
    # A compound type operand is carried through rather than refused as a
    # malformed `ascribe#`: only a ground type is written as a bare symbol, and
    # a binder may be ascribed any type the type-level grammar admits.
    #
    # The name remains the only binder of the enclosing form. The operand's
    # scope covers the whole of it, so both constructor identifiers carry
    # {961}; neither is resolved, since a constructor names no variable.
    expanded = expand(
        read("(lambda ((ascribe# x (@list @int))) x)"), fresh=counter(960)
    )

    binder = Symbol("x", {Scope(960)})
    assert expanded == SList(
        [
            Symbol("lambda"),
            SList(
                [
                    SList(
                        [
                            Symbol("ascribe#"),
                            binder,
                            SList(
                                [
                                    Symbol("@list", {Scope(961)}),
                                    Symbol("@int", {Scope(961)}),
                                ]
                            ),
                        ]
                    )
                ]
            ),
            binder,
        ]
    )


def test_a_quantifier_scopes_its_variable_over_its_body():
    # `forall#` is a binding form, so it mints a scope of its own exactly as
    # `lambda` does. Three scopes are in play: {961} is the lambda's, reaching
    # the name and nothing else; {962} is the type operand's, covering the whole
    # written type; {963} is the quantifier's, covering its body, which the
    # bound variable joins by union.
    #
    # That is what lets the parse phase resolve a type variable through the one
    # binding mechanism rather than carry a set of identifiers beside the table.
    expanded = expand(
        read("(lambda ((ascribe# f (forall# (a) (@fun (@prod a) a)))) f)"),
        fresh=counter(961),
    )

    binder = Symbol("f", {Scope(961)})
    quantified = Symbol("a", {Scope(962), Scope(963)})
    body = {Scope(962), Scope(963)}
    assert expanded == SList(
        [
            Symbol("lambda"),
            SList(
                [
                    SList(
                        [
                            Symbol("ascribe#"),
                            binder,
                            SList(
                                [
                                    Symbol("forall#", {Scope(962)}),
                                    SList([quantified]),
                                    SList(
                                        [
                                            Symbol("@fun", body),
                                            SList([Symbol("@prod", body), quantified]),
                                            quantified,
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    )
                ]
            ),
            binder,
        ]
    )


def test_anonymous_symbols_are_refused():
    with pytest.raises(ExpandError):
        expand(Symbol(""), frozenset({Scope(940)}))


def test_an_ascribed_lambda_parameter_scopes_its_name_and_its_type_apart():
    # (lambda ((ascribe# x @int)) x) scopes `x` exactly as a bare parameter
    # would, at {951}. The type operand is a region of its own and takes the
    # next scope, {952} — which is what keeps a type variable bound inside it
    # from ever being the same binder as a term variable of one identifier.
    #
    # `@int` is a constructor rather than a variable, so nothing resolves it;
    # it carries the operand's scope only because the scope covers the whole
    # operand.
    expanded = expand(read("(lambda ((ascribe# x @int)) x)"), fresh=counter(951))

    binder = Symbol("x", {Scope(951)})
    assert expanded == SList(
        [
            Symbol("lambda"),
            SList([SList([Symbol("ascribe#"), binder, Symbol("@int", {Scope(952)})])]),
            binder,
        ]
    )


def test_an_ascribed_letrec_binder_scopes_its_name_and_its_type_apart():
    # The letrec region covers the initializer as well as the body, so both
    # carry {952}; the type operand takes its own scope, {953}, as it does in a
    # lambda parameter.
    expanded = expand(read("(letrec (((ascribe# n @int) 5)) n)"), fresh=counter(952))

    binder = Symbol("n", {Scope(952)})
    assert expanded == SList(
        [
            Symbol("letrec"),
            SList(
                [
                    SList(
                        [
                            SList(
                                [
                                    Symbol("ascribe#"),
                                    binder,
                                    Symbol("@int", {Scope(953)}),
                                ]
                            ),
                            Integer(5),
                        ]
                    )
                ]
            ),
            binder,
        ]
    )


def test_an_ascribed_binder_resolves_like_a_bare_one():
    # The ascription is not a binding form of its own: the name it wraps is
    # registered in the table exactly as a bare parameter is, so the body's
    # occurrence resolves to it.
    expanded = expand(read("(lambda ((ascribe# x @int)) x)"), fresh=counter(953))
    table = bindings(expanded)

    assert resolve(Symbol("x", {Scope(953)}), table).binder == Symbol("x", {Scope(953)})


def test_an_ascribed_type_operand_is_not_a_binding():
    # The type spelling is registered nowhere, so resolving it finds nothing
    # — which is what keeps it from colliding with a program identifier that
    # happens to share its spelling.
    expanded = expand(read("(lambda ((ascribe# x @int)) x)"), fresh=counter(954))

    with pytest.raises(Unbound):
        resolve(Symbol("@int", {Scope(954)}), bindings(expanded))


# -- define as a core form -------------------------------------------------------


def test_a_define_binder_mints_no_scope():
    # A declaration is visible to the whole program, so its binder takes no
    # scope of its own: one minted here would be absent from every later
    # top-level form's references, and none of them could resolve the name.
    expanded = expand(read("(define x 5)"), fresh=counter(960))

    assert expanded == SList([Symbol("define"), Symbol("x"), Integer(5)])
    assert resolve(Symbol("x"), bindings(expanded)).binder == Symbol("x")


def test_an_ascribed_define_binder_keeps_its_wrapper():
    # A declaration mints no scope for its binder, so the type operand's is the
    # first this expansion mints: the name stays bare and the operand takes
    # {961}.
    expanded = expand(read("(define (ascribe# f @int) 5)"), fresh=counter(961))

    assert expanded == SList(
        [
            Symbol("define"),
            SList([Symbol("ascribe#"), Symbol("f"), Symbol("@int", {Scope(961)})]),
            Integer(5),
        ]
    )


def test_a_define_binder_is_registered_before_its_initializer():
    # Registering the binder first is what lets a defined procedure refer to
    # itself.
    expanded = expand(read("(define f (lambda (x) (f x)))"), fresh=counter(962))
    table = bindings(expanded)

    assert resolve(Symbol("f"), table).binder == Symbol("f")


def test_a_nested_define_is_refused():
    # `expand_expr` has no notion of depth, so a define below the top level
    # would register a binder carrying the enclosing form's scopes — one
    # indistinguishable from that form's own binder.
    with pytest.raises(ExpandError):
        expand(read("(lambda (x) (define x 5))"), fresh=counter(963))

    with pytest.raises(ExpandError):
        expand(read("(match 1 ((#t) (define y 5)) ((#f) 2))"), fresh=counter(964))


def test_an_ascribed_macro_name_is_refused():
    # A macro name binds a transformer, which is not a value and has no type.
    source = "(letrec-syntax (((ascribe# m @int) (syntax-rules () ((m) 1)))) (m))"

    with pytest.raises(ExpandError):
        expand(read(source), fresh=counter(965))


def test_an_ascribed_pattern_variable_is_refused():
    # A pattern variable binds matched syntax, not a value. A template may
    # still construct an ascription, which
    # `test_a_macro_introduced_ascription_is_recognized` covers.
    source = "(letrec-syntax ((m (syntax-rules () ((m (ascribe# v @int)) v)))) (m 1))"

    with pytest.raises(ExpandError):
        expand(read(source), fresh=counter(966))


# -- type binders ----------------------------------------------------------------


def test_a_recursive_type_scopes_its_variable_over_its_body():
    # `@mu` binds exactly as `forall#` does, with one scope for the form: {971}
    # is the type operand's and {972} the `@mu`'s, which its body carries and
    # its binder joins by union.
    expanded = expand(
        read("(lambda ((ascribe# xs (@mu a (@list a)))) xs)"), fresh=counter(970)
    )

    binder = Symbol("xs", {Scope(970)})
    bound = Symbol("a", {Scope(971), Scope(972)})
    assert expanded == SList(
        [
            Symbol("lambda"),
            SList(
                [
                    SList(
                        [
                            Symbol("ascribe#"),
                            binder,
                            SList(
                                [
                                    Symbol("@mu", {Scope(971)}),
                                    bound,
                                    SList(
                                        [
                                            Symbol("@list", {Scope(971), Scope(972)}),
                                            bound,
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    )
                ]
            ),
            binder,
        ]
    )


def test_nested_type_binders_of_one_identifier_are_distinct():
    # The payoff of scoping type variables: an inner binder reusing an outer
    # identifier is a distinct binder, told apart by its scope set rather than
    # by overwriting an entry in a set of identifiers. The occurrence carries
    # both, so `resolve` takes the inner one — and the outer's set is also a
    # subset of the inner's, which keeps the candidates a chain and the choice
    # unambiguous.
    expanded = expand(
        read("(lambda ((ascribe# x (@mu a (@prod (@mu a (@prod a)))))) x)"),
        fresh=counter(975),
    )

    match expanded:
        case SList([_, SList([SList([_, _, outer_mu])]), _]):
            pass
        case _:
            raise AssertionError(f"not an ascribed lambda: {expanded}")

    match outer_mu:
        case SList([_, outer, SList([_, inner_mu])]):
            pass
        case _:
            raise AssertionError(f"not an outer @mu: {outer_mu}")

    match inner_mu:
        case SList([_, inner, SList([_, occurrence])]):
            pass
        case _:
            raise AssertionError(f"not an inner @mu: {inner_mu}")

    assert outer.identifier == inner.identifier == "a"
    assert outer != inner
    assert outer.scopes < inner.scopes
    assert inner.scopes <= occurrence.scopes


@pytest.mark.parametrize(
    "source",
    [
        # A quantifier binding one identifier twice: both binding sites would
        # share the quantifier's single scope and so be indistinguishable, which
        # is the refusal `checked` makes of a lambda's parameters.
        "(lambda ((ascribe# f (forall# (a a) a))) f)",
        # Every `@` identifier is a type constructor's, so one names no
        # variable. Refused before any binder is registered for it.
        "(lambda ((ascribe# x (@mu @int (@list @int)))) x)",
        "(lambda ((ascribe# f (forall# (@int) @int))) f)",
        # A type variable is minted through the one minting path, so it takes
        # the refusals a term binder takes — including the core-form
        # identifiers, which no type needs.
        "(lambda ((ascribe# x (@mu lambda (@list lambda)))) x)",
        # A binder position that is not a symbol at all.
        "(lambda ((ascribe# x (@mu (a) (@list a)))) x)",
    ],
)
def test_malformed_type_binders_are_refused(source: str):
    with pytest.raises(ExpandError):
        expand(read(source))


# -- free variables --------------------------------------------------------------


def test_forall_mints_a_scopeless_binder():
    # The form is the binding position and the only occurrence of what it
    # binds, so it opens no region and mints no scope. Its binder is free in
    # the whole program and carries nothing. The type operand is a region of
    # its own, and takes the one scope every ascribed type takes.
    expanded = expand(read("(forall (ascribe# n @int))"), fresh=counter(980))

    assert expanded == SList(
        [
            Symbol("forall"),
            SList(
                [
                    Symbol("ascribe#"),
                    Symbol("n"),
                    Symbol("@int", {Scope(980)}),
                ]
            ),
        ]
    )
    assert list(bindings(expanded)) == [Symbol("n")]


def test_two_occurrences_at_two_depths_mint_the_one_binder():
    # The scopes a written occurrence has acquired are the enclosing regions',
    # and keeping them would make these two occurrences binders with
    # incomparable scope sets: two variables where the program wrote one, and
    # an ambiguous reference to either. Both mint the bare `n`.
    source = read(
        "(lt (forall (ascribe# n @int)) (lambda (x) (forall (ascribe# n @int))))"
    )
    table = bindings(expand(source, fresh=counter(985)))

    assert [b for b in table if b.identifier == "n"] == [Symbol("n")]

    # So a reference reaches it from inside the lambda as readily as outside.
    assert resolve(Symbol("n", {Scope(985)}), table).binder == Symbol("n")
    assert resolve(Symbol("n"), table).binder == Symbol("n")


def test_a_free_variable_must_be_ascribed():
    # Nothing encloses the binder, so there is nowhere else its type could
    # have been written.
    with pytest.raises(ExpandError):
        expand(read("(forall n)"), fresh=counter(988))


def test_a_malformed_forall_is_refused():
    for source in ["(forall)", "(forall (ascribe# n @int) 1)", "(forall (ascribe# n))"]:
        with pytest.raises(ExpandError):
            expand(read(source), fresh=counter(990))


def test_forall_cannot_be_shadowed():
    # A keyword, so no binder may take its spelling and no macro may be bound
    # to it — the refusal every core form shares.
    with pytest.raises(ExpandError):
        expand(read("(lambda (forall) forall)"), fresh=counter(995))
