from collections.abc import Callable
from itertools import count

import pytest

from Expander import expand, parse
from Reader import read
from Resolve import BindingTable, bindings
from Syntax import Expr, Scope
from Syntax.Type import BOOL, CharType, IntType, Type, list_of
from TypeCheck import TypeChecker, TypeCheckError, check_toplevel

# Counters from the 1800s, so no expectation here depends on the global fresh-uid
# counter and no uid collides with another test file's.


def counter(start: int) -> Callable[[], Scope]:
    uids = count(start)
    return lambda: Scope(next(uids))


def program_of(source: str, expand_at: int) -> tuple[Expr, BindingTable]:
    expanded = expand(read(source), fresh=counter(expand_at))
    program = parse(expanded, bindings(expanded))
    return program, bindings(program)


def checker_of(source: str, expand_at: int) -> tuple[Expr, TypeChecker]:
    program, table = program_of(source, expand_at)
    return program, TypeChecker(table=table)


# -- the type a form already has -------------------------------------------------


def test_a_literal_has_its_own_type():
    for source, ty in [("5", IntType()), ("#\\a", CharType())]:
        program, checker = checker_of(source, 1800)

        assert checker.type_of(program) == ty


def test_a_reference_has_its_ascription():
    # Read through a letrec, because an ascription is what puts a type in the
    # store and a bare reference has nowhere to have been ascribed.
    program, checker = checker_of("(letrec (((ascribe# n @int) 5)) n)", 1801)

    assert checker.type_of(program) == IntType()


def test_an_unascribed_reference_has_no_type():
    program, checker = checker_of("(letrec ((n 5)) n)", 1802)

    with pytest.raises(TypeCheckError):
        checker.type_of(program)


def test_an_application_has_its_heads_return_type():
    source = (
        "(letrec (((ascribe# f (@fun (@prod @int) @bool))"
        " (lambda (x) (#t (prod#)))))"
        " (f 1))"
    )
    program, checker = checker_of(source, 1803)

    assert checker.type_of(program) == BOOL


def test_a_polymorphic_head_is_instantiated_from_its_operands():
    # `len` names its instance in no return type, so this is the reading that
    # `solved` adds: `a` comes from the operand.
    source = "(letrec (((ascribe# xs (@list @int)) '(1 2))) (len xs))"
    program, checker = checker_of(source, 1804)

    assert checker.type_of(program) == IntType()


def test_a_match_takes_its_first_clauses_type():
    program, checker = checker_of(
        "(match (le 1 2) ((#t (prod#)) 1) ((#f (prod#)) 2))", 1805
    )

    assert checker.type_of(program) == IntType()


def test_a_quoted_datum_takes_the_type_of_its_first_element():
    program, checker = checker_of("'(1 2 3)", 1806)

    assert checker.type_of(program) == list_of(IntType())


def test_a_quoted_empty_list_determines_no_type():
    # It inhabits every list type and names none of them.
    program, checker = checker_of("'()", 1807)

    with pytest.raises(TypeCheckError):
        checker.type_of(program)


def test_a_quoted_symbol_determines_no_type():
    program, checker = checker_of("'a", 1808)

    with pytest.raises(TypeCheckError):
        checker.type_of(program)


def test_a_lambda_has_no_type_of_its_own():
    # A lambda's type is what a binder's ascription supplies, and this one has
    # no binder.
    program, checker = checker_of("(lambda ((ascribe# x @int)) x)", 1809)

    with pytest.raises(TypeCheckError):
        checker.type_of(program)


# -- quoted data checked ---------------------------------------------------------


def test_a_quoted_list_of_integers_inhabits_a_list_of_integers():
    program, checker = checker_of("'(1 2 3)", 1810)

    checker.check(program, list_of(IntType()))


def test_a_quoted_empty_list_inhabits_every_list_type():
    for element in [IntType(), CharType(), list_of(IntType())]:
        program, checker = checker_of("'()", 1811)

        checker.check(program, list_of(element))


def test_a_quoted_list_is_refused_where_an_element_does_not_inhabit():
    program, checker = checker_of("'(1 #\\a)", 1812)

    with pytest.raises(TypeCheckError):
        checker.check(program, list_of(IntType()))


def test_a_quoted_list_does_not_inhabit_the_type_of_its_elements():
    program, checker = checker_of("'(1 2)", 1813)

    with pytest.raises(TypeCheckError):
        checker.check(program, IntType())


def test_a_quoted_list_of_lists_inhabits_a_list_of_lists():
    program, checker = checker_of("'((1) (2 3))", 1814)

    checker.check(program, list_of(list_of(IntType())))


def test_a_quoted_symbol_inhabits_nothing():
    for ty in [IntType(), BOOL, list_of(IntType())]:
        program, checker = checker_of("'a", 1815)

        with pytest.raises(TypeCheckError):
            checker.check(program, ty)


# -- instantiation from the operands ---------------------------------------------


def listed(source: str, expand_at: int) -> tuple[Expr, TypeChecker]:
    # `source` checked with two list-of-integer binders in scope, which is the
    # shape both `eq` and `len` are applied at in `lib/`.
    return checker_of(
        "(letrec (((ascribe# xs (@list @int)) '(1 2))"
        "         ((ascribe# ys (@list @int)) '(3)))"
        f" {source})",
        expand_at,
    )


def test_eq_compares_two_lists():
    program, checker = listed("(eq xs ys)", 1816)

    checker.check(program, BOOL)


def test_eq_compares_a_list_against_the_empty_list():
    # The operand determines `a`, and the quoted empty list is then checked at
    # it — which is why the `quote` arm and `solved` arrive together.
    program, checker = listed("(eq xs '())", 1817)

    checker.check(program, BOOL)


def test_eq_compares_two_integers():
    program, checker = checker_of("(eq 1 2)", 1818)

    checker.check(program, BOOL)


def test_eq_is_refused_where_its_operands_disagree():
    # `a` is taken at `@int` from the first operand, and the second is refused
    # at that instance.
    program, checker = checker_of("(eq 1 (#t (prod#)))", 1819)

    with pytest.raises(TypeCheckError):
        checker.check(program, BOOL)


def test_len_measures_a_list():
    program, checker = listed("(len xs)", 1820)

    checker.check(program, IntType())


def test_len_is_refused_on_what_is_not_a_list():
    program, checker = checker_of("(len 5)", 1821)

    with pytest.raises(TypeCheckError):
        checker.check(program, IntType())


def test_eq_is_refused_where_no_operand_determines_its_instance():
    # Both operands are empty lists, so nothing names the element type and the
    # refusal is `instantiated`'s, unchanged.
    program, checker = checker_of("(eq '() '())", 1822)

    with pytest.raises(TypeCheckError):
        checker.check(program, BOOL)


# -- check_toplevel --------------------------------------------------------------


def test_a_top_level_declaration_is_checked():
    program, checker = checker_of("(define (ascribe# n @int) 5)", 1823)

    check_toplevel(program, checker)


def test_a_top_level_declaration_is_refused_at_a_type_it_does_not_inhabit():
    program, checker = checker_of("(define (ascribe# n @int) (#t (prod#)))", 1824)

    with pytest.raises(TypeCheckError):
        check_toplevel(program, checker)


def test_a_top_level_expression_is_checked_at_the_type_it_has():
    program, checker = checker_of(
        "(match (le 1 2) ((#t (prod#)) 1) ((#f (prod#)) 2))", 1825
    )

    check_toplevel(program, checker)


def test_a_top_level_expression_is_refused_where_its_branches_disagree():
    program, checker = checker_of(
        "(match (le 1 2) ((#t (prod#)) 1) ((#f (prod#)) #\\a))", 1826
    )

    with pytest.raises(TypeCheckError):
        check_toplevel(program, checker)


def test_a_declaration_is_visible_to_a_later_form():
    # What the carried store is for: the ascription of a `define` outlives the
    # form that wrote it, so a later form's application of it checks.
    declaration, table = program_of(
        "(define (ascribe# double (@fun (@prod @int) @int)) (lambda (x) (+ x x)))",
        1827,
    )
    checker = TypeChecker(table=table)
    check_toplevel(declaration, checker)

    use, uses = program_of("(double 2)", 1828)
    later = TypeChecker(table=uses, store=checker.store)

    assert later.type_of(use) == IntType()
    check_toplevel(use, later)


def test_a_reference_to_no_declaration_is_refused():
    program, checker = checker_of("(double 2)", 1829)

    with pytest.raises(TypeCheckError):
        check_toplevel(program, checker)


def test_a_top_level_lambda_is_refused():
    # Not a type error in the lambda: there is no binder to read its type off.
    program, checker = checker_of("(lambda ((ascribe# x @int)) x)", 1830)

    with pytest.raises(TypeCheckError):
        check_toplevel(program, checker)


def test_a_top_level_letrec_is_checked_at_its_bodys_type():
    source = "(letrec (((ascribe# n @int) 5)) n)"
    program, checker = checker_of(source, 1831)
    ty: Type = checker.type_of(program)

    assert ty == IntType()
    check_toplevel(program, TypeChecker(table=bindings(program)))
