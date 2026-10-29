from collections.abc import Callable
from itertools import count

import pytest

from Expander import ExpandError, expand
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


def test_if_introduces_no_scope():
    # The branches stay in the enclosing region: every occurrence carries
    # exactly the lambda's scope, and the keyword carries the same context.
    expanded = expand(read("(lambda (x) (if x x x))"), fresh=counter(920))

    occurrence = Symbol("x", {Scope(920)})
    expected_body = SList(
        [Symbol("if", {Scope(920)}), occurrence, occurrence, occurrence]
    )
    assert expanded == SList([Symbol("lambda"), SList([occurrence]), expected_body])


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


def test_choice_branches_expand_in_the_enclosing_region():
    # choice binds nothing and introduces nothing: it is handled as a plain
    # application, so the keyword and both branches acquire the enclosing
    # context while the numerals pass through.
    expanded = expand(read("(lambda (x) (choice 1 2 x x))"), fresh=counter(930))

    occurrence = Symbol("x", {Scope(930)})
    assert expanded == SList(
        [
            Symbol("lambda"),
            SList([occurrence]),
            SList(
                [
                    Symbol("choice", {Scope(930)}),
                    Integer(1),
                    Integer(2),
                    occurrence,
                    occurrence,
                ]
            ),
        ]
    )


@pytest.mark.parametrize(
    "source",
    [
        "(lambda (x x) x)",
        "(letrec ((f f) (f f)) f)",
        "(lambda (if) if)",
        "(letrec ((lambda (lambda (x) x))) x)",
        "(lambda x x)",
        "(lambda (x))",
        "(if x x)",
        "(letrec ((f)) f)",
        "(letrec (f) f)",
        "(lambda ((ascribe# x)) x)",
        "(letrec (((ascribe# n) 5)) n)",
        "(lambda ((ascribe# x @int @int)) x)",
        "(lambda ((ascribe# x (@list @int))) x)",
        "(ascribe# x @int)",
        "(lambda (ascribe#) ascribe#)",
        "(letrec ((ascribe# 5)) ascribe#)",
        "(define)",
        "(define x)",
        "(define x 1 2)",
        "(lambda (define) define)",
        "(letrec ((define 5)) define)",
    ],
)
def test_malformed_core_forms_are_refused(source: str):
    with pytest.raises(ExpandError):
        expand(read(source))


def test_anonymous_symbols_are_refused():
    with pytest.raises(ExpandError):
        expand(Symbol(""), frozenset({Scope(940)}))


def test_an_ascribed_lambda_parameter_scopes_only_its_name():
    # (lambda ((ascribe# x @int)) x) scopes `x` exactly as a bare parameter
    # would, at {951}. The type operand is not a binder and not a reference,
    # so it stays bare: it denotes a type, and nothing resolves it.
    expanded = expand(read("(lambda ((ascribe# x @int)) x)"), fresh=counter(951))

    binder = Symbol("x", {Scope(951)})
    assert expanded == SList(
        [
            Symbol("lambda"),
            SList([SList([Symbol("ascribe#"), binder, Symbol("@int")])]),
            binder,
        ]
    )


def test_an_ascribed_letrec_binder_scopes_only_its_name():
    # The letrec region covers the initializer as well as the body, so both
    # carry {952}; the type operand stays bare there too.
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
                                    Symbol("@int"),
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
    expanded = expand(read("(define (ascribe# f @int) 5)"), fresh=counter(961))

    assert expanded == SList(
        [
            Symbol("define"),
            SList([Symbol("ascribe#"), Symbol("f"), Symbol("@int")]),
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
        expand(read("(if 1 (define y 5) 2)"), fresh=counter(964))


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
