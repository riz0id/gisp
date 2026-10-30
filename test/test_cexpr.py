from collections.abc import Callable
from itertools import count

from Analysis.CExpr import (
    CExpr,
    CharCExpr,
    ComparisonCExpr,
    ConjunctionCExpr,
    DisjunctionCExpr,
    EquivalenceCExpr,
    IntegerCExpr,
    ListCExpr,
    NegationCExpr,
    VarCExpr,
)
from Expander import expand, parse
from Reader import read
from Resolve import BindingTable, bindings
from Syntax import Char, Expr, Integer, Scope, SList, Symbol


def counter() -> Callable[[], Scope]:
    # Every program under test mints its own uids from zero, so the scopes in
    # an expectation are a function of that program alone and never of the
    # process-wide counter `expand` draws from by default.
    uids = count()
    return lambda: Scope(next(uids))


def program(source: str) -> Expr:
    expanded = expand(read(source), fresh=counter())
    return parse(expanded, bindings(expanded))


def body(source: str) -> tuple[Expr, BindingTable]:
    # The body of a binding form, with the store of the whole program it came
    # from: a store rebuilt from the body alone cannot see the binders of the
    # form enclosing it, so a reference to one would resolve as free.
    whole: Expr = program(source)

    match whole:
        case SList([Symbol(), SList(), inner]):
            return inner, bindings(whole)
        case _:
            raise AssertionError(f"not a binding form: {whole}")


# -- the built-in forms --------------------------------------------------------


def test_a_conjunction_is_built_from_an_and_form():
    expr = program("(and #t #f)")

    assert CExpr.from_expr(expr, bindings(expr)) == ConjunctionCExpr(
        VarCExpr(Symbol("#t")), VarCExpr(Symbol("#f"))
    )


def test_a_disjunction_is_built_from_an_or_form():
    expr = program("(or #t #f)")

    assert CExpr.from_expr(expr, bindings(expr)) == DisjunctionCExpr(
        VarCExpr(Symbol("#t")), VarCExpr(Symbol("#f"))
    )


def test_a_comparison_is_built_from_an_lt_form():
    expr = program("(lt 1 2)")

    assert CExpr.from_expr(expr, bindings(expr)) == ComparisonCExpr(
        IntegerCExpr(Integer(1)), IntegerCExpr(Integer(2))
    )


def test_an_equivalence_is_built_from_an_eq_form():
    expr = program("(eq 1 2)")

    assert CExpr.from_expr(expr, bindings(expr)) == EquivalenceCExpr(
        IntegerCExpr(Integer(1)), IntegerCExpr(Integer(2))
    )


def test_a_negation_is_built_from_a_not_form():
    expr = program("(not #t)")

    assert CExpr.from_expr(expr, bindings(expr)) == NegationCExpr(
        VarCExpr(Symbol("#t"))
    )


def test_operands_are_converted_too():
    expr, store = body("(lambda (n) (and (lt n 1) #t))")

    assert CExpr.from_expr(expr, store) == ConjunctionCExpr(
        ComparisonCExpr(VarCExpr(Symbol("n", {Scope(0)})), IntegerCExpr(Integer(1))),
        VarCExpr(Symbol("#t")),
    )


def test_a_negated_operand_is_converted_too():
    expr, store = body("(lambda (n) (not (lt n 1)))")

    assert CExpr.from_expr(expr, store) == NegationCExpr(
        ComparisonCExpr(VarCExpr(Symbol("n", {Scope(0)})), IntegerCExpr(Integer(1)))
    )


def test_a_built_in_is_recognized_with_no_store_of_its_own():
    # The five operators are in the store this conversion builds, so a caller
    # with nothing to say about bindings still gets the forms.
    expr = SList([Symbol("and"), Symbol("#t"), Symbol("#f")])

    assert CExpr.from_expr(expr) == ConjunctionCExpr(
        VarCExpr(Symbol("#t")), VarCExpr(Symbol("#f"))
    )


# -- when a built-in form is not one -------------------------------------------


def test_a_shadowed_built_in_is_an_application():
    expr, store = body("(lambda (and) (and #t #f))")

    # The parameter outranks the bare built-in binder, so the head names a
    # variable and the form is the application it looks like.
    assert CExpr.from_expr(expr, store) == ListCExpr(
        [Symbol("and", {Scope(0)}), Symbol("#t"), Symbol("#f")]
    )


def test_a_letrec_bound_built_in_is_an_application():
    expr, store = body("(letrec ((eq 1)) (eq 2 3))")

    assert CExpr.from_expr(expr, store) == ListCExpr(
        [Symbol("eq", {Scope(0)}), Integer(2), Integer(3)]
    )


def test_a_shadowed_negation_is_an_application():
    expr, store = body("(lambda (not) (not #t))")

    assert CExpr.from_expr(expr, store) == ListCExpr(
        [Symbol("not", {Scope(0)}), Symbol("#t")]
    )


def test_an_over_applied_built_in_is_an_application():
    # Each built-in form has one arity, so a longer list is not one of them.
    expr = program("(and #t #f #t)")

    assert CExpr.from_expr(expr, bindings(expr)) == ListCExpr(
        [Symbol("and"), Symbol("#t"), Symbol("#f"), Symbol("#t")]
    )


def test_an_under_applied_built_in_is_an_application():
    expr = program("(and #t)")

    assert CExpr.from_expr(expr, bindings(expr)) == ListCExpr(
        [Symbol("and"), Symbol("#t")]
    )


def test_an_over_applied_negation_is_an_application():
    expr = program("(not #t #f)")

    assert CExpr.from_expr(expr, bindings(expr)) == ListCExpr(
        [Symbol("not"), Symbol("#t"), Symbol("#f")]
    )


def test_an_under_applied_negation_is_an_application():
    expr = program("(not)")

    assert CExpr.from_expr(expr, bindings(expr)) == ListCExpr([Symbol("not")])


def test_an_unbound_head_is_an_application():
    expr = program("(f #t #f)")

    assert CExpr.from_expr(expr, bindings(expr)) == ListCExpr(
        [Symbol("f"), Symbol("#t"), Symbol("#f")]
    )


def test_an_application_keeps_its_operands_as_syntax():
    # `ListCExpr` holds `Expr`, so conversion stops at the first head that is
    # not a built-in: the comparison inside is left as the syntax it was.
    expr = program("(f (lt 1 2))")

    assert CExpr.from_expr(expr, bindings(expr)) == ListCExpr(
        [Symbol("f"), SList([Symbol("lt"), Integer(1), Integer(2)])]
    )


def test_the_empty_list_is_an_application():
    assert CExpr.from_expr(SList([])) == ListCExpr([])


# -- leaves --------------------------------------------------------------------


def test_an_integer_is_an_integer_literal():
    assert CExpr.from_expr(Integer(1)) == IntegerCExpr(Integer(1))


def test_a_character_is_a_character_literal():
    assert CExpr.from_expr(Char(ord("a"))) == CharCExpr(Char(ord("a")))


def test_a_symbol_is_a_variable():
    assert CExpr.from_expr(Symbol("x", {Scope(0)})) == VarCExpr(Symbol("x", {Scope(0)}))


# -- round trips ---------------------------------------------------------------


def test_the_built_in_forms_round_trip():
    for source in ["(and #t #f)", "(or #t #f)", "(lt 1 2)", "(eq 1 2)", "(not #t)"]:
        expr = program(source)

        assert CExpr.from_expr(expr, bindings(expr)).to_expr() == expr


def test_a_form_with_a_variable_round_trips():
    expr, store = body("(lambda (n) (lt n 1))")

    assert CExpr.from_expr(expr, store).to_expr() == expr


def test_the_leaves_round_trip():
    for expr in [Integer(1), Char(ord("a")), Symbol("x")]:
        assert CExpr.from_expr(expr).to_expr() == expr


# -- the store -----------------------------------------------------------------


def test_the_callers_store_is_left_alone():
    # The built-in binders are added to a copy: a caller's store is shared
    # structure, and a conversion has no business putting binders in it.
    expr, store = body("(lambda (n) (and (lt n 1) #t))")
    before: int = len(store)

    CExpr.from_expr(expr, store)

    assert len(store) == before
    assert Symbol("and") not in store


def test_a_free_variable_converts_to_the_variable_it_binds():
    # A `forall` is an occurrence of what it binds, and the condition language
    # reads it as that variable. Its reading has to precede the unary operator
    # arm's: the form is itself a two-item list, and an application of `forall`
    # is a condition nothing can read.
    guard = program("(lt (forall (ascribe# n @int)) 5)")

    assert CExpr.from_expr(guard) == ComparisonCExpr(
        VarCExpr(Symbol("n")),
        IntegerCExpr(Integer(5)),
    )
