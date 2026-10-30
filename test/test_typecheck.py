from collections.abc import Callable
from itertools import count

import pytest
from hypothesis import given
from hypothesis import strategies as st

from Expander import ParseError, expand, parse
from Reader import read
from Resolve import BindingTable, bindings
from Syntax import Expr, Integer, Scope, Symbol
from Syntax.Type import (
    BOOL,
    CharType,
    Context,
    FunType,
    IntType,
    MuType,
    ProdType,
    SumType,
    Type,
    VariadicStarType,
    VariadicType,
    VarType,
    list_of,
)
from TypeCheck import TypeChecker, TypeCheckError

# Fixtures inject deterministic counters from the 1400s so no expectation
# depends on the global fresh-uid counter and no uid collides with another
# test file's hand-written scopes.


def counter(start: int) -> Callable[[], Scope]:
    uids = count(start)
    return lambda: Scope(next(uids))


def program_of(source: str, expand_at: int) -> tuple[Expr, BindingTable]:
    expanded = expand(read(source), fresh=counter(expand_at))
    program = parse(expanded, bindings(expanded))
    return program, bindings(program)


def var(identifier: str, *uids: int) -> Symbol:
    return Symbol(identifier, {Scope(uid) for uid in uids})


def unary(return_type: Type, operand_type: Type) -> FunType:
    return FunType(
        return_type=return_type,
        argument_type=ProdType({Symbol("a"): operand_type}),
    )


# -- literals --------------------------------------------------------------------


def test_integer_literals_check_at_int():
    program, table = program_of("5", 1400)
    TypeChecker(table=table).check(program, IntType())


def test_char_literals_check_at_char():
    program, table = program_of(r"#\a", 1401)
    TypeChecker(table=table).check(program, CharType())


def test_literals_refuse_foreign_types():
    program, table = program_of("5", 1402)
    for ty in [CharType(), BOOL, list_of(IntType())]:
        with pytest.raises(TypeCheckError):
            TypeChecker(table=table).check(program, ty)


def test_the_boolean_constructors_check_at_bool():
    # `@bool` is a sum, so a boolean is written the way every sum's values are:
    # the constructor applied to its payload, which here is the nullary
    # product.
    for source in ["(#t (prod#))", "(#f (prod#))"]:
        program, table = program_of(source, 1403)
        TypeChecker(table=table).check(program, BOOL)
        with pytest.raises(TypeCheckError):
            TypeChecker(table=table).check(program, IntType())


def test_a_bare_boolean_symbol_is_not_an_expression():
    # The special case that is gone: nothing ascribes `#t`, so it is an
    # unascribed reference like any other. `(#t)` is refused too, for its
    # arity — a sum constructor takes its payload.
    for source in ["#t", "#f", "(#t)", "(#f)"]:
        program, table = program_of(source, 1404)

        with pytest.raises(TypeCheckError):
            TypeChecker(table=table).check(program, BOOL)


def test_the_sum_written_out_accepts_the_boolean_constructors():
    # `@bool` written as the sum it is accepts exactly the same terms, which is
    # what it means for the two to be one type rather than two.
    program, table = program_of("(#t (prod#))", 1405)
    written = SumType({Symbol("#t"): ProdType(), Symbol("#f"): ProdType()})

    assert written == BOOL
    TypeChecker(table=table).check(program, written)


def test_only_the_boolean_constructors_construct_bool():
    # `is_constructor` reads a boolean head by resolving it against the sum's
    # keys, so a binder of `@bool` type is an ordinary reference in head
    # position and not a constructor of it.
    program, table = program_of("(lambda ((ascribe# b @bool)) (b 1))", 1406)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, unary(IntType(), BOOL))


# -- variables -------------------------------------------------------------------


def test_an_ascribed_binder_checks_at_its_ascription():
    program, table = program_of("(lambda (x) x)", 1405)

    TypeChecker(table=table).check(program, unary(IntType(), IntType()))


def test_a_reference_off_its_ascription_is_refused():
    program, table = program_of("(lambda (x) x)", 1406)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, unary(BOOL, IntType()))


def test_an_unascribed_reference_is_refused():
    program, table = program_of("(lambda (x) y)", 1407)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, unary(IntType(), IntType()))


# -- lambda ----------------------------------------------------------------------


def test_a_parameter_count_off_the_product_is_refused():
    program, table = program_of("(lambda (x y) x)", 1410)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, unary(IntType(), IntType()))


def test_a_lambda_does_not_inhabit_a_variadic_function_type():
    program, table = program_of("(lambda (x) x)", 1411)
    ty = FunType(return_type=IntType(), argument_type=VariadicType(IntType()))

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ty)


def test_a_lambda_does_not_inhabit_a_ground_type():
    program, table = program_of("(lambda (x) x)", 1412)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, IntType())


# -- match -----------------------------------------------------------------------


def test_a_match_constrains_the_scrutinee_and_every_clause_body():
    program, table = program_of("(lambda (x) (match (le x 1) ((#t) x) ((#f) 5)))", 1415)

    TypeChecker(table=table).check(program, unary(IntType(), IntType()))


def test_a_non_boolean_scrutinee_is_refused():
    # `(#t)` is not a constructor of `@int`, which is what refuses this: the
    # scrutinee's own type is what a pattern is read against.
    program, table = program_of("(lambda (x) (match 1 ((#t) x) ((#f) x)))", 1416)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, unary(IntType(), IntType()))


def test_a_clause_body_off_the_checked_type_is_refused():
    program, table = program_of(
        "(lambda (x) (match (le x 1) ((#t) x) ((#f) (#f (prod#)))))", 1417
    )

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, unary(IntType(), IntType()))


def test_an_ill_typed_scrutinee_is_refused():
    # `type_of` answers an application's return type without looking at its
    # operands, so a scrutinee has to be checked at what was read as well as
    # read — or `(le #t #t)` would pass for the `@bool` it claims to be.
    program, table = program_of(
        "(lambda (x) (match (le (#t (prod#)) (#t (prod#))) ((#t) x) ((#f) x)))", 1418
    )

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, unary(IntType(), IntType()))


def test_a_list_match_covers_both_constructors():
    program, table = program_of("(match xs ((nil) 0) ((cons y ys) 1))", 1420)
    store = Context.default()
    store.register(Symbol("xs"), list_of(IntType()))

    TypeChecker(table=table, store=store).check(program, IntType())


def test_a_list_match_missing_a_constructor_is_refused():
    program, table = program_of("(match xs ((nil) 0))", 1421)
    store = Context.default()
    store.register(Symbol("xs"), list_of(IntType()))

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table, store=store).check(program, IntType())


def test_a_nested_pattern_does_not_cover_its_constructor_on_its_own():
    # Naming both constructors is not coverage: this one misses every list of
    # two or more, so exhaustiveness is a question about a constructor's fields
    # and not about the set of names a clause mentions.
    program, table = program_of("(match xs ((nil) 0) ((cons y (nil)) 1))", 1422)
    store = Context.default()
    store.register(Symbol("xs"), list_of(IntType()))

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table, store=store).check(program, IntType())

    # Adding the clause that does cover it makes the same three-way shape the
    # libraries are written in.
    program, table = program_of(
        "(match xs ((nil) 0) ((cons y (nil)) 1) ((cons y ys) 2))", 1423
    )
    store = Context.default()
    store.register(Symbol("xs"), list_of(IntType()))

    TypeChecker(table=table, store=store).check(program, IntType())


def test_a_pattern_binds_its_operands_at_the_payload_types():
    # `y` is the element and `ys` the rest, which is the whole of what the cons
    # payload says — and the body is what proves the checker registered them.
    program, table = program_of("(match xs ((nil) xs) ((cons y ys) ys))", 1424)
    store = Context.default()
    store.register(Symbol("xs"), list_of(IntType()))

    TypeChecker(table=table, store=store).check(program, list_of(IntType()))

    # The element is an `@int`, so a clause returning it at a list type is not.
    program, table = program_of("(match xs ((nil) xs) ((cons y ys) y))", 1425)
    store = Context.default()
    store.register(Symbol("xs"), list_of(IntType()))

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table, store=store).check(program, list_of(IntType()))


def test_a_pattern_operand_count_off_the_payload_is_refused():
    program, table = program_of("(match xs ((nil) 0) ((cons y) 1))", 1426)
    store = Context.default()
    store.register(Symbol("xs"), list_of(IntType()))

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table, store=store).check(program, IntType())


def test_an_integer_match_is_exhaustive_only_with_an_irrefutable_clause():
    store = Context.default()
    store.register(Symbol("n"), IntType())

    program, table = program_of("(match n (0 1) (m m))", 1427)
    TypeChecker(table=table, store=store).check(program, IntType())

    # A literal covers one value of `@int`, and `@int` has a constructor per
    # value — so no run of literals is ever exhaustive.
    program, table = program_of("(match n (0 1))", 1428)
    store = Context.default()
    store.register(Symbol("n"), IntType())

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table, store=store).check(program, IntType())


def test_a_procedure_cannot_be_matched_on():
    # Nothing takes a closure apart: `lambda` builds one, `eq` raises on one,
    # and a pattern over one would have no field to name.
    program, table = program_of("(match f ((lambda) 1) (g 2))", 1429)
    store = Context.default()
    store.register(Symbol("f"), unary(IntType(), IntType()))

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table, store=store).check(program, IntType())


def test_a_sum_is_taken_apart_by_its_constructors():
    # The eliminator sums never had. `check` could already build one, and this
    # is the first thing that reads one back.
    program, table = program_of("(match s ((zero) 0) ((succ n) n))", 1430)
    store = Context.default()
    store.register(
        Symbol("s"),
        SumType({Symbol("zero"): ProdType(), Symbol("succ"): IntType()}),
    )

    TypeChecker(table=table, store=store).check(program, IntType())


def test_a_sum_match_missing_a_constructor_is_refused():
    program, table = program_of("(match s ((succ n) n))", 1431)
    store = Context.default()
    store.register(
        Symbol("s"),
        SumType({Symbol("zero"): ProdType(), Symbol("succ"): IntType()}),
    )

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table, store=store).check(program, IntType())


def test_a_product_match_is_irrefutable_and_names_the_fields():
    # One constructor, so a product pattern refutes nothing — and it is the
    # only way to project a field, since the language has no selectors.
    program, table = program_of("(match p ((prod# a b) a))", 1432)
    store = Context.default()
    store.register(
        Symbol("p"),
        ProdType({Symbol("x"): IntType(), Symbol("y"): BOOL}),
    )

    TypeChecker(table=table, store=store).check(program, IntType())

    program, table = program_of("(match p ((prod# a b) b))", 1433)
    store = Context.default()
    store.register(
        Symbol("p"),
        ProdType({Symbol("x"): IntType(), Symbol("y"): BOOL}),
    )

    TypeChecker(table=table, store=store).check(program, BOOL)


def test_a_recursive_sum_is_matched_through_its_unfolding():
    # `(@mu a ...)` is taken apart as its unfolding, so the recursive position
    # comes back as the folded type and a pattern over it is written at the
    # type the program wrote.
    nat = MuType(
        Symbol("a"),
        SumType({Symbol("zero"): ProdType(), Symbol("succ"): VarType(Symbol("a"))}),
    )
    program, table = program_of("(match s ((zero) s) ((succ m) m))", 1434)
    store = Context.default()
    store.register(Symbol("s"), nat)

    TypeChecker(table=table, store=store).check(program, nat)


def test_a_pattern_binder_takes_an_ascription_checked_against_its_position():
    program, table = program_of(
        "(match xs ((nil) 0) ((cons (ascribe# y @int) ys) y))", 1435
    )
    store = Context.default()
    store.register(Symbol("xs"), list_of(IntType()))

    TypeChecker(table=table, store=store).check(program, IntType())

    # Ascribed off its position, which the function type fixes.
    program, table = program_of(
        "(match xs ((nil) 0) ((cons (ascribe# y @bool) ys) 0))", 1436
    )
    store = Context.default()
    store.register(Symbol("xs"), list_of(IntType()))

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table, store=store).check(program, IntType())


def test_two_clauses_may_bind_the_one_spelling():
    # Each clause is a region of its own, so the two `y` binders are distinct
    # symbols and the store holds both.
    program, table = program_of("(match xs ((cons y ys) y) (y 0))", 1437)
    store = Context.default()
    store.register(Symbol("xs"), list_of(IntType()))

    TypeChecker(table=table, store=store).check(program, IntType())


# -- letrec ----------------------------------------------------------------------


def test_letrec_checks_initializers_against_their_ascriptions():
    program, table = program_of("(letrec ((n 5)) n)", 1420)
    store: Context = Context.default()
    store.register(var("n", 1420), IntType())

    TypeChecker(table=table, store=store).check(program, IntType())


def test_an_unascribed_letrec_binder_is_refused():
    program, table = program_of("(letrec ((n 5)) n)", 1421)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, IntType())


def test_an_initializer_off_its_ascription_is_refused():
    program, table = program_of("(letrec ((n 5)) n)", 1422)
    store: Context = Context.default()
    store.register(var("n", 1422), BOOL)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table, store=store).check(program, BOOL)


# -- applications ----------------------------------------------------------------


def test_builtin_arithmetic_applications_check():
    program, table = program_of("(lambda (x) (+ x 1))", 1425)

    TypeChecker(table=table).check(program, unary(IntType(), IntType()))


def test_builtin_comparison_applications_check():
    program, table = program_of("(lambda (x) (lt x 1))", 1426)

    TypeChecker(table=table).check(program, unary(BOOL, IntType()))


def test_an_application_off_its_return_type_is_refused():
    program, table = program_of("(lambda (x) (+ x 1))", 1427)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, unary(BOOL, IntType()))


def test_an_operand_off_its_argument_type_is_refused():
    program, table = program_of("(lambda (x) (+ x (#t (prod#))))", 1428)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, unary(IntType(), IntType()))


def test_an_operand_count_off_the_product_is_refused():
    program, table = program_of("(+ 1 2 3)", 1429)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, IntType())


def test_an_unascribed_application_head_is_refused():
    program, table = program_of("(len 5)", 1430)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, IntType())


def test_a_non_symbol_application_head_is_refused():
    program, table = program_of("((lambda (x) x) 5)", 1431)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, IntType())


# -- variadic applications --------------------------------------------------------


def test_variadic_applications_admit_any_argument_count():
    store: Context = Context.default()
    store.register(
        Symbol("f"),
        FunType(return_type=BOOL, argument_type=VariadicType(IntType())),
    )

    for source in ["(f)", "(f 1)", "(f 1 2 3)"]:
        program, table = program_of(source, 1435)
        TypeChecker(table=table, store=store).check(program, BOOL)


def test_variadic_operands_share_the_one_type():
    store: Context = Context.default()
    store.register(
        Symbol("f"),
        FunType(return_type=BOOL, argument_type=VariadicType(IntType())),
    )
    program, table = program_of("(f 1 (#t (prod#)))", 1436)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table, store=store).check(program, BOOL)


def test_a_variadic_star_run_requires_an_argument():
    store: Context = Context.default()
    store.register(
        Symbol("g"),
        FunType(return_type=BOOL, argument_type=VariadicStarType(IntType())),
    )

    program, table = program_of("(g 1)", 1437)
    TypeChecker(table=table, store=store).check(program, BOOL)

    program, table = program_of("(g)", 1438)
    with pytest.raises(TypeCheckError):
        TypeChecker(table=table, store=store).check(program, BOOL)


# -- constructions ----------------------------------------------------------------


def test_a_cell_construction_checks_its_payload_fieldwise():
    # `(@list t)` is a sum, so a cell is built the way a sum's values are: the
    # constructor applied to its payload, which here is the two-field record
    # `(@prod (x t) (xs (@list t)))`.
    program, table = program_of("(cons (prod# 1 (nil (prod#))))", 1440)

    TypeChecker(table=table).check(program, list_of(IntType()))


def test_a_list_element_off_the_element_type_is_refused():
    program, table = program_of("(cons (prod# (#t (prod#)) (nil (prod#))))", 1441)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, list_of(IntType()))


def test_the_empty_list_construction_checks_at_any_element_type():
    program, table = program_of("(nil (prod#))", 1442)

    TypeChecker(table=table).check(program, list_of(IntType()))
    TypeChecker(table=table).check(program, list_of(BOOL))


def test_a_cell_payload_is_not_spread_across_the_operands():
    # The asymmetry every sum has: a pattern spreads the payload's fields, so
    # `(cons x xs)` takes a cell apart with two operands, while a construction
    # applies the constructor to the one payload. `(#t (prod#))` is written for
    # the same reason.
    program, table = program_of("(cons 1 (nil (prod#)))", 1443)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, list_of(IntType()))


def test_neither_list_nor_dot_constructs_a_list():
    # Both were removed with the sort. `list` was the one n-ary constructor the
    # language had, and a sum constructor takes a single payload; `.` stood in
    # for the cell constructor the sort had no name for, and the sum names it
    # `cons`. Neither is ascribed or bound, so each is refused as the
    # unascribed application head it now is.
    for source, at in [("(list 1 2 3)", 1444), ("(. 1 (nil (prod#)))", 1445)]:
        program, table = program_of(source, at)

        with pytest.raises(TypeCheckError):
            TypeChecker(table=table).check(program, list_of(IntType()))


def test_product_constructions_check_fieldwise():
    program, table = program_of("(prod# 1 (#t (prod#)))", 1445)
    ty = ProdType({Symbol("a"): IntType(), Symbol("b"): BOOL})

    TypeChecker(table=table).check(program, ty)


def test_a_field_count_off_the_product_is_refused():
    program, table = program_of("(prod# 1)", 1446)
    ty = ProdType({Symbol("a"): IntType(), Symbol("b"): BOOL})

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ty)


def test_sum_constructions_check_their_payload():
    program, table = program_of("(c 5)", 1450)
    ty = SumType({Symbol("c"): IntType()})

    TypeChecker(table=table).check(program, ty)

    program, table = program_of("(c (#t (prod#)))", 1451)
    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ty)


# -- shadowing --------------------------------------------------------------------


def test_a_shadowed_constructor_identifier_is_an_ordinary_application():
    # The binder named `cons` captures the body's head, so the application
    # reads through the binder's ascription rather than constructing a list.
    program, table = program_of("(lambda (cons) (cons 1))", 1455)
    ty = FunType(
        return_type=IntType(),
        argument_type=ProdType({Symbol("f"): unary(IntType(), IntType())}),
    )

    TypeChecker(table=table).check(program, ty)


def test_a_shadowed_cons_identifier_does_not_construct_a_list():
    program, table = program_of("(lambda (cons) (cons 1))", 1456)
    ty = FunType(
        return_type=list_of(IntType()),
        argument_type=ProdType({Symbol("f"): unary(IntType(), IntType())}),
    )

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ty)


# -- list selectors ---------------------------------------------------------------

# `head` and `tail` are ascribed schemes in `Context.default()` rather than
# special-cased. Each writes its quantified variable into its result, so the
# checked type names the instance and nothing has to be synthesized.


def test_head_checks_its_operand_at_a_list_of_the_checked_type():
    program, table = program_of("(head '(1 2))", 1460)

    TypeChecker(table=table).check(program, IntType())


def test_head_reads_the_element_type_off_the_checked_type():
    # A quoted datum cannot stand in here: no datum is a boolean, so the list
    # is built from its constructors.
    program, table = program_of(
        "(head (cons (prod# (#t (prod#)) (nil (prod#)))))", 1461
    )

    TypeChecker(table=table).check(program, BOOL)
    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, IntType())


def test_head_refuses_an_operand_that_is_not_a_list():
    program, table = program_of("(head 5)", 1462)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, IntType())


def test_tail_checks_at_the_list_type_it_returns():
    program, table = program_of("(tail '(1 2))", 1463)

    TypeChecker(table=table).check(program, list_of(IntType()))


def test_tail_does_not_return_an_element():
    program, table = program_of("(tail '(1))", 1464)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, IntType())


def test_the_selectors_compose():
    program, table = program_of("(head (tail '(1 2)))", 1465)

    TypeChecker(table=table).check(program, IntType())


def test_a_selector_operand_reads_through_its_ascription():
    program, table = program_of("(lambda (xs) (head xs))", 1466)

    TypeChecker(table=table).check(program, unary(IntType(), list_of(IntType())))


def test_the_selectors_take_one_argument():
    program, table = program_of("(head '(1) '(2))", 1467)
    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, IntType())

    program, table = program_of("(tail)", 1468)
    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, list_of(IntType()))


def test_a_shadowed_selector_identifier_is_an_ordinary_application():
    # The binder named `head` captures the body's head, so the application
    # reads through the binder's ascription rather than selecting from a list.
    # A shadowing binder is a scoped symbol, which is in neither namespace of
    # the store until `abstraction` registers it.
    program, table = program_of("(lambda (head) (head 1))", 1469)
    ty = FunType(
        return_type=IntType(),
        argument_type=ProdType({Symbol("f"): unary(IntType(), IntType())}),
    )

    TypeChecker(table=table).check(program, ty)


# -- polymorphism: instantiation from the checked type ----------------------------


def test_a_selector_instantiates_at_several_element_types():
    # `head`'s scheme writes its variable into its return type, so the checked
    # type is what names the instance — and the new `@mu` is what `solve` has
    # to read it through.
    for source, element in [
        ("(head '(1 2))", IntType()),
        ("(head (cons (prod# (#t (prod#)) (nil (prod#)))))", BOOL),
        (r"(head '(#\a #\b))", CharType()),
        ("(head '((1) (2)))", list_of(IntType())),
    ]:
        program, table = program_of(source, 1600)
        TypeChecker(table=table).check(program, element)


def test_a_cell_construction_checks_at_several_element_types():
    for source, element in [
        ("(cons (prod# 1 (nil (prod#))))", IntType()),
        ("(cons (prod# (#t (prod#)) (nil (prod#))))", BOOL),
        (r"(cons (prod# #\a (nil (prod#))))", CharType()),
    ]:
        program, table = program_of(source, 1601)
        TypeChecker(table=table).check(program, list_of(element))


def test_a_cell_refuses_an_element_off_the_list():
    program, table = program_of("(cons (prod# 1 (nil (prod#))))", 1602)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, list_of(BOOL))


def test_a_cell_builds_a_list_of_lists():
    # The payload is read off the checked type whatever its sort, so a compound
    # element type is taken as readily as a ground one.
    program, table = program_of("(cons (prod# '(1) (nil (prod#))))", 1603)

    TypeChecker(table=table).check(program, list_of(list_of(IntType())))


def test_a_polymorphic_builtin_is_not_checkable_as_a_reference():
    # A scheme is instantiated where its binder is applied, so naming one
    # without applying it is refused — at the function type it would have at an
    # instance as readily as at any other.
    program, table = program_of("head", 1604)
    ty = FunType(
        return_type=IntType(),
        argument_type=ProdType([list_of(IntType())]),
    )

    for checked in [ty, IntType()]:
        with pytest.raises(TypeCheckError):
            TypeChecker(table=table).check(program, checked)


# -- polymorphism: compound and quantified ascriptions ----------------------------


def test_a_binder_can_be_ascribed_a_compound_type():
    program, table = program_of("(letrec (((ascribe# xs (@list @int)) '(1))) xs)", 1640)

    TypeChecker(table=table).check(program, list_of(IntType()))


def test_a_compound_ascription_off_its_initializer_is_refused():
    program, table = program_of(
        "(letrec (((ascribe# xs (@list @int))"
        " (cons (prod# (#t (prod#)) (nil (prod#)))))) xs)",
        1641,
    )

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, list_of(IntType()))


def test_a_parameter_cannot_be_ascribed_a_quantified_type():
    # A parameter's type is fixed by the position it sits in, so a quantifier
    # over it would bind variables no use could instantiate.
    source = "(lambda ((ascribe# f (forall# (a) (@fun (@prod a) a)))) f)"
    program, table = program_of(source, 1642)
    ty = unary(unary(IntType(), IntType()), unary(IntType(), IntType()))

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ty)


def test_a_declared_polymorphic_binder_is_used_at_two_instances():
    # The headline behavior: one `define` checked once, used at two different
    # instances. The body is checked against the scheme's body with `a` abstract,
    # so the one check stands for every instance a use reads off.
    identity = "(define (ascribe# id (forall# (a) (@fun (@prod a) a))) (lambda (x) x))"
    program, table = program_of(identity, 1643)
    checker = TypeChecker(table=table)
    checker.declaration(program)

    for source, ty in [("(id 1)", IntType()), ("(id (#t (prod#)))", BOOL)]:
        use, uses = program_of(source, 1644)
        TypeChecker(table=uses, store=checker.store).check(use, ty)


def test_a_polymorphic_body_cannot_assume_an_instance():
    # `a` is abstract while the body is checked, so nothing constructs it: a
    # literal returned from a function claiming to return any `a` is refused.
    source = "(define (ascribe# f (forall# (a) (@fun (@prod a) @int))) (lambda (x) x))"
    program, table = program_of(source, 1645)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).declaration(program)


def test_a_literal_does_not_inhabit_an_abstract_variable():
    # The crash this guards: `is_constructor` answers `False` for an abstract
    # variable rather than resolving it, so this is a `TypeCheckError` and not a
    # `ValueError` escaping from the type queries.
    source = "(define (ascribe# f (forall# (a) (@fun (@prod a) a))) (lambda (x) 5))"
    program, table = program_of(source, 1646)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).declaration(program)


def test_a_polymorphic_declaration_passes_its_variable_through():
    # What a polymorphic body *can* do: return its argument, and apply another
    # scheme at the abstract variable.
    source = (
        "(define (ascribe# first (forall# (a) (@fun (@prod (@list a)) a)))"
        " (lambda (xs) (head xs)))"
    )
    program, table = program_of(source, 1647)
    checker = TypeChecker(table=table)
    checker.declaration(program)

    use, uses = program_of("(first '(1 2))", 1648)
    TypeChecker(table=uses, store=checker.store).check(use, IntType())


# -- tuple products ---------------------------------------------------------------

# A tuple is a product whose fields are anonymous, so each case here is the
# counterpart of a record case above with the names taken away.


def test_tuple_constructions_check_positionally():
    program, table = program_of("(prod# 1 (#t (prod#)))", 1470)

    TypeChecker(table=table).check(program, ProdType([IntType(), BOOL]))

    program, table = program_of("(prod# 1 (#t (prod#)))", 1470)
    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ProdType([BOOL, IntType()]))


def test_a_field_count_off_a_tuple_is_refused():
    program, table = program_of("(prod# 1)", 1471)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ProdType([IntType(), BOOL]))


def test_a_lambda_inhabits_a_function_type_over_a_tuple():
    program, table = program_of("(lambda (x) x)", 1472)
    ty = FunType(return_type=IntType(), argument_type=ProdType([IntType()]))

    TypeChecker(table=table).check(program, ty)

    program, table = program_of("(lambda (x y) x)", 1472)
    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ty)


def test_an_application_through_a_tuple_argument_type_checks_its_operands():
    program, table = program_of("(lambda (f) (f 1))", 1473)
    operator = FunType(return_type=IntType(), argument_type=ProdType([IntType()]))
    ty = FunType(return_type=IntType(), argument_type=ProdType([operator]))

    TypeChecker(table=table).check(program, ty)

    program, table = program_of("(lambda (f) (f (#t (prod#))))", 1473)
    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ty)


# -- laws -------------------------------------------------------------------------


@given(st.integers())
def test_every_integer_checks_at_int_and_nothing_else_ground(value: int):
    expr = Integer(value)

    TypeChecker().check(expr, IntType())
    for ty in [CharType(), BOOL]:
        with pytest.raises(TypeCheckError):
            TypeChecker().check(expr, ty)


# -- type ascription -------------------------------------------------------------


def test_an_ascribed_letrec_binder_reads_its_type_from_source():
    # The headline behavior: no external `store.register` call. Every other
    # letrec test above must pre-ascribe the binder from Python, because
    # check-only has nowhere to synthesize a type from — the ascription is
    # where a type written in source reaches the store.
    program, table = program_of("(letrec (((ascribe# n @int) 5)) n)", 1480)

    TypeChecker(table=table).check(program, IntType())


def test_an_ascribed_letrec_initializer_off_its_ascription_is_refused():
    program, table = program_of("(letrec (((ascribe# n @int) (#t (prod#)))) n)", 1481)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, IntType())


def test_an_ascribed_letrec_binder_checks_its_body_at_the_ascription():
    # The body's occurrence resolves to the ascribed binder, so checking the
    # whole form at a foreign type fails on the reference.
    program, table = program_of("(letrec (((ascribe# n @int) 5)) n)", 1482)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, BOOL)


def test_an_ascription_contradicting_a_standing_one_is_refused():
    # `Context.register` refuses a binder already ascribed something else, and
    # `binding` surfaces that as a type error rather than letting it escape.
    program, table = program_of("(letrec (((ascribe# n @int) 5)) n)", 1483)
    store: Context = Context.default()
    store.register(var("n", 1483), BOOL)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table, store=store).check(program, IntType())


def test_an_ascription_agreeing_with_a_standing_one_is_accepted():
    program, table = program_of("(letrec (((ascribe# n @int) 5)) n)", 1484)
    store: Context = Context.default()
    store.register(var("n", 1484), IntType())

    TypeChecker(table=table, store=store).check(program, IntType())


def test_an_ascribed_lambda_parameter_agreeing_with_its_signature_checks():
    program, table = program_of("(lambda ((ascribe# x @int)) x)", 1485)

    TypeChecker(table=table).check(program, unary(IntType(), IntType()))


def test_an_ascribed_lambda_parameter_off_its_signature_is_refused():
    # Unlike a letrec binder, a parameter's type is already fixed by the
    # function type, so the ascription is checked against it, not read for it.
    program, table = program_of("(lambda ((ascribe# x @bool)) x)", 1486)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, unary(IntType(), IntType()))


def test_every_nullary_type_can_be_ascribed():
    for spelling, ty in [("@int", IntType()), ("@bool", BOOL)]:
        program, table = program_of(f"(lambda ((ascribe# x {spelling})) x)", 1487)
        TypeChecker(table=table).check(program, unary(ty, ty))


def test_an_unrecognized_ascribed_type_is_refused():
    # `@list` is a type constructor, and a constructor is written as a list:
    # `(@list @int)` is a type and a bare `@list` is not. That is a question
    # about written form, so the parse phase refuses it and the checker is never
    # handed the program — where this once arrived as a type error.
    with pytest.raises(ParseError):
        program_of("(letrec (((ascribe# n @list) 5)) n)", 1488)


def test_an_ascribed_char_parameter_checks_at_char():
    program, table = program_of("(lambda ((ascribe# c @char)) c)", 1489)

    TypeChecker(table=table).check(program, unary(CharType(), CharType()))


# -- declarations ----------------------------------------------------------------


def test_an_ascribed_define_reads_its_type_from_source():
    program, table = program_of("(define (ascribe# n @int) 5)", 1490)

    TypeChecker(table=table).declaration(program)


def test_an_ascribed_define_initializer_off_its_ascription_is_refused():
    program, table = program_of("(define (ascribe# n @int) (#t (prod#)))", 1491)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).declaration(program)


def test_an_unascribed_define_is_refused():
    # Check-only has nowhere to synthesize a declaration's type from.
    program, table = program_of("(define n 5)", 1492)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).declaration(program)


def test_an_unascribed_define_checks_at_a_standing_ascription():
    program, table = program_of("(define n 5)", 1493)
    store: Context = Context.default()
    store.register(Symbol("n"), IntType())

    TypeChecker(table=table, store=store).declaration(program)


def test_a_define_ascription_contradicting_a_standing_one_is_refused():
    program, table = program_of("(define (ascribe# n @int) 5)", 1494)
    store: Context = Context.default()
    store.register(Symbol("n"), BOOL)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table, store=store).declaration(program)


def test_a_define_is_not_a_checkable_expression():
    # A declaration is not an expression. Refused outright rather than
    # falling into the application arm, which would misreport it as a call to
    # an unascribed `define`.
    program, table = program_of("(define (ascribe# n @int) 5)", 1495)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, IntType())


def test_a_declaration_refuses_a_non_define():
    program, table = program_of("5", 1496)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).declaration(program)


# -- free variables --------------------------------------------------------------


def test_a_free_variable_checks_at_the_type_it_ranges_over():
    for name, written, ty in [
        ("n", "@int", IntType()),
        ("c", "@char", CharType()),
    ]:
        program, table = program_of(f"(forall (ascribe# {name} {written}))", 1480)
        checker = TypeChecker(table=table)

        checker.check(program, ty)

        # Registered on sight: the form is the binding position as well as the
        # occurrence, so nothing else would ever carry the ascription in.
        assert checker.store[Symbol(name)] == ty


def test_a_free_variable_over_bool_is_refused():
    # `@bool` is a sum, and a sum is not a ground type. A free variable stands
    # for a range of values the analysis takes it over, and only a type written
    # as a bare symbol has one — which `@bool` is written as and no longer is.
    program, table = program_of("(forall (ascribe# b @bool))", 1482)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, BOOL)


def test_a_free_variable_refuses_a_foreign_type():
    program, table = program_of("(forall (ascribe# n @int))", 1481)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, BOOL)


def test_a_free_variable_answers_its_own_type():
    # `type_of` is what lets the form stand at the top level and in an operand
    # position that determines a scheme's instance.
    program, table = program_of("(forall (ascribe# n @int))", 1482)

    assert TypeChecker(table=table).type_of(program) == IntType()


def test_only_a_ground_type_has_values_to_range_over():
    # A `Support` is a set of integers, so only a nullary type's values are a
    # range: list data has no bound and a procedure is not a value to quantify
    # across.
    for written in ["(@list @int)", "(@fun (@prod @int) @int)", "(@variadic @int)"]:
        program, table = program_of(f"(forall (ascribe# v {written}))", 1483)

        with pytest.raises(TypeCheckError):
            TypeChecker(table=table).type_of(program)


def test_a_free_variable_refuses_a_polymorphic_ascription():
    # The refusal a parameter takes: the variable stands for values of the one
    # type it is written at, so a quantifier would bind what no occurrence
    # instantiates.
    program, table = program_of(
        "(forall (ascribe# n (forall# (a) @int)))",
        1484,
    )

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).type_of(program)


def test_two_occurrences_of_one_free_variable_agree():
    program, table = program_of(
        "(lt (forall (ascribe# n @int)) (forall (ascribe# n @int)))",
        1485,
    )
    checker = TypeChecker(table=table)

    checker.check(program, BOOL)

    assert checker.store[Symbol("n")] == IntType()


def test_two_occurrences_at_two_types_are_refused():
    # Both occurrences name the one variable, so the second ascription is a
    # rebinding and `Context.register` refuses it.
    program, table = program_of(
        "(lt (forall (ascribe# n @int)) (forall (ascribe# n @bool)))",
        1486,
    )

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, BOOL)
