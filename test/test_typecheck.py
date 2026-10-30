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
    BoolType,
    CharType,
    Context,
    FunType,
    IntType,
    ListType,
    ProdType,
    SumType,
    Type,
    VariadicStarType,
    VariadicType,
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
    for ty in [CharType(), BoolType(), ListType(IntType())]:
        with pytest.raises(TypeCheckError):
            TypeChecker(table=table).check(program, ty)


def test_the_boolean_symbols_check_at_bool():
    for source in ["#t", "#f"]:
        program, table = program_of(source, 1403)
        TypeChecker(table=table).check(program, BoolType())
        with pytest.raises(TypeCheckError):
            TypeChecker(table=table).check(program, IntType())


# -- variables -------------------------------------------------------------------


def test_an_ascribed_binder_checks_at_its_ascription():
    program, table = program_of("(lambda (x) x)", 1405)

    TypeChecker(table=table).check(program, unary(IntType(), IntType()))


def test_a_reference_off_its_ascription_is_refused():
    program, table = program_of("(lambda (x) x)", 1406)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, unary(BoolType(), IntType()))


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


# -- if --------------------------------------------------------------------------


def test_if_constrains_the_test_and_both_branches():
    program, table = program_of("(lambda (x) (if #t x 5))", 1415)

    TypeChecker(table=table).check(program, unary(IntType(), IntType()))


def test_a_non_boolean_test_is_refused():
    program, table = program_of("(lambda (x) (if 1 x x))", 1416)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, unary(IntType(), IntType()))


def test_a_branch_off_the_checked_type_is_refused():
    program, table = program_of("(lambda (x) (if #t x #f))", 1417)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, unary(IntType(), IntType()))


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
    store.register(var("n", 1422), BoolType())

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table, store=store).check(program, BoolType())


# -- applications ----------------------------------------------------------------


def test_builtin_arithmetic_applications_check():
    program, table = program_of("(lambda (x) (+ x 1))", 1425)

    TypeChecker(table=table).check(program, unary(IntType(), IntType()))


def test_builtin_comparison_applications_check():
    program, table = program_of("(lambda (x) (lt x 1))", 1426)

    TypeChecker(table=table).check(program, unary(BoolType(), IntType()))


def test_an_application_off_its_return_type_is_refused():
    program, table = program_of("(lambda (x) (+ x 1))", 1427)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, unary(BoolType(), IntType()))


def test_an_operand_off_its_argument_type_is_refused():
    program, table = program_of("(lambda (x) (+ x #t))", 1428)

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
        FunType(return_type=BoolType(), argument_type=VariadicType(IntType())),
    )

    for source in ["(f)", "(f 1)", "(f 1 2 3)"]:
        program, table = program_of(source, 1435)
        TypeChecker(table=table, store=store).check(program, BoolType())


def test_variadic_operands_share_the_one_type():
    store: Context = Context.default()
    store.register(
        Symbol("f"),
        FunType(return_type=BoolType(), argument_type=VariadicType(IntType())),
    )
    program, table = program_of("(f 1 #t)", 1436)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table, store=store).check(program, BoolType())


def test_a_variadic_star_run_requires_an_argument():
    store: Context = Context.default()
    store.register(
        Symbol("g"),
        FunType(return_type=BoolType(), argument_type=VariadicStarType(IntType())),
    )

    program, table = program_of("(g 1)", 1437)
    TypeChecker(table=table, store=store).check(program, BoolType())

    program, table = program_of("(g)", 1438)
    with pytest.raises(TypeCheckError):
        TypeChecker(table=table, store=store).check(program, BoolType())


# -- constructions ----------------------------------------------------------------


def test_list_constructions_check_elementwise():
    program, table = program_of("(list 1 2 3)", 1440)

    TypeChecker(table=table).check(program, ListType(IntType()))


def test_a_list_element_off_the_element_type_is_refused():
    program, table = program_of("(list 1 #t)", 1441)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ListType(IntType()))


def test_the_empty_list_construction_checks_at_any_element_type():
    program, table = program_of("(list)", 1442)

    TypeChecker(table=table).check(program, ListType(IntType()))
    TypeChecker(table=table).check(program, ListType(BoolType()))


def test_product_constructions_check_fieldwise():
    program, table = program_of("(prod# 1 #t)", 1445)
    ty = ProdType({Symbol("a"): IntType(), Symbol("b"): BoolType()})

    TypeChecker(table=table).check(program, ty)


def test_a_field_count_off_the_product_is_refused():
    program, table = program_of("(prod# 1)", 1446)
    ty = ProdType({Symbol("a"): IntType(), Symbol("b"): BoolType()})

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ty)


def test_sum_constructions_check_their_payload():
    program, table = program_of("(c 5)", 1450)
    ty = SumType({Symbol("c"): IntType()})

    TypeChecker(table=table).check(program, ty)

    program, table = program_of("(c #t)", 1451)
    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ty)


# -- shadowing --------------------------------------------------------------------


def test_a_shadowed_constructor_spelling_is_an_ordinary_application():
    # The binder spelled `list` captures the body's head, so the application
    # reads through the binder's ascription rather than constructing a list.
    program, table = program_of("(lambda (list) (list 1))", 1455)
    ty = FunType(
        return_type=IntType(),
        argument_type=ProdType({Symbol("f"): unary(IntType(), IntType())}),
    )

    TypeChecker(table=table).check(program, ty)


def test_a_shadowed_list_spelling_does_not_construct_a_list():
    program, table = program_of("(lambda (list) (list 1))", 1456)
    ty = FunType(
        return_type=ListType(IntType()),
        argument_type=ProdType({Symbol("f"): unary(IntType(), IntType())}),
    )

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ty)


# -- list selectors ---------------------------------------------------------------

# `head` and `tail` are ascribed schemes in `Context.default()` rather than
# special-cased. Each writes its quantified variable into its result, so the
# checked type names the instance and nothing has to be synthesized.


def test_head_checks_its_operand_at_a_list_of_the_checked_type():
    program, table = program_of("(head (list 1 2))", 1460)

    TypeChecker(table=table).check(program, IntType())


def test_head_reads_the_element_type_off_the_checked_type():
    program, table = program_of("(head (list #t))", 1461)

    TypeChecker(table=table).check(program, BoolType())
    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, IntType())


def test_head_refuses_an_operand_that_is_not_a_list():
    program, table = program_of("(head 5)", 1462)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, IntType())


def test_tail_checks_at_the_list_type_it_returns():
    program, table = program_of("(tail (list 1 2))", 1463)

    TypeChecker(table=table).check(program, ListType(IntType()))


def test_tail_does_not_return_an_element():
    program, table = program_of("(tail (list 1))", 1464)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, IntType())


def test_the_selectors_compose():
    program, table = program_of("(head (tail (list 1 2)))", 1465)

    TypeChecker(table=table).check(program, IntType())


def test_a_selector_operand_reads_through_its_ascription():
    program, table = program_of("(lambda (xs) (head xs))", 1466)

    TypeChecker(table=table).check(program, unary(IntType(), ListType(IntType())))


def test_the_selectors_take_one_argument():
    program, table = program_of("(head (list 1) (list 2))", 1467)
    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, IntType())

    program, table = program_of("(tail)", 1468)
    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ListType(IntType()))


def test_a_shadowed_selector_spelling_is_an_ordinary_application():
    # The binder spelled `head` captures the body's head, so the application
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


def test_cons_instantiates_at_several_element_types():
    for source, element in [
        ("(. 1 (list 2))", IntType()),
        ("(. #t (list #f))", BoolType()),
        (r"(. #\a (list #\b))", CharType()),
    ]:
        program, table = program_of(source, 1600)
        TypeChecker(table=table).check(program, ListType(element))


def test_cons_refuses_an_element_off_the_list():
    program, table = program_of("(. 1 (list #t))", 1601)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ListType(IntType()))


def test_cons_builds_a_list_of_lists():
    # The instance is read off the checked type whatever its sort, so a compound
    # one instantiates as readily as a ground one.
    program, table = program_of("(. (list 1) (list (list 2)))", 1602)

    TypeChecker(table=table).check(program, ListType(ListType(IntType())))


def test_a_polymorphic_builtin_is_not_checkable_as_a_reference():
    # A scheme is instantiated where its binder is applied, so naming one
    # without applying it is refused — at the function type it would have at an
    # instance as readily as at any other.
    program, table = program_of("head", 1603)
    ty = FunType(
        return_type=IntType(),
        argument_type=ProdType([ListType(IntType())]),
    )

    for checked in [ty, IntType()]:
        with pytest.raises(TypeCheckError):
            TypeChecker(table=table).check(program, checked)


# -- polymorphism: compound and quantified ascriptions ----------------------------


def test_a_binder_can_be_ascribed_a_compound_type():
    program, table = program_of(
        "(letrec (((ascribe# xs (@list @int)) (list 1))) xs)", 1640
    )

    TypeChecker(table=table).check(program, ListType(IntType()))


def test_a_compound_ascription_off_its_initializer_is_refused():
    program, table = program_of(
        "(letrec (((ascribe# xs (@list @int)) (list #t))) xs)", 1641
    )

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ListType(IntType()))


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

    for source, ty in [("(id 1)", IntType()), ("(id #t)", BoolType())]:
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

    use, uses = program_of("(first (list 1 2))", 1648)
    TypeChecker(table=uses, store=checker.store).check(use, IntType())


# -- tuple products ---------------------------------------------------------------

# A tuple is a product whose fields are anonymous, so each case here is the
# counterpart of a record case above with the names taken away.


def test_tuple_constructions_check_positionally():
    program, table = program_of("(prod# 1 #t)", 1470)

    TypeChecker(table=table).check(program, ProdType([IntType(), BoolType()]))

    program, table = program_of("(prod# 1 #t)", 1470)
    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ProdType([BoolType(), IntType()]))


def test_a_field_count_off_a_tuple_is_refused():
    program, table = program_of("(prod# 1)", 1471)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ProdType([IntType(), BoolType()]))


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

    program, table = program_of("(lambda (f) (f #t))", 1473)
    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ty)


# -- laws -------------------------------------------------------------------------


@given(st.integers())
def test_every_integer_checks_at_int_and_nothing_else_ground(value: int):
    expr = Integer(value)

    TypeChecker().check(expr, IntType())
    for ty in [CharType(), BoolType()]:
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
    program, table = program_of("(letrec (((ascribe# n @int) #t)) n)", 1481)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, IntType())


def test_an_ascribed_letrec_binder_checks_its_body_at_the_ascription():
    # The body's occurrence resolves to the ascribed binder, so checking the
    # whole form at a foreign type fails on the reference.
    program, table = program_of("(letrec (((ascribe# n @int) 5)) n)", 1482)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, BoolType())


def test_an_ascription_contradicting_a_standing_one_is_refused():
    # `Context.register` refuses a binder already ascribed something else, and
    # `binding` surfaces that as a type error rather than letting it escape.
    program, table = program_of("(letrec (((ascribe# n @int) 5)) n)", 1483)
    store: Context = Context.default()
    store.register(var("n", 1483), BoolType())

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
    for spelling, ty in [("@int", IntType()), ("@bool", BoolType())]:
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
    program, table = program_of("(define (ascribe# n @int) #t)", 1491)

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
    store.register(Symbol("n"), BoolType())

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
