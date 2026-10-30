from collections.abc import Callable
from itertools import count

from Analysis import conditions
from Analysis.CExpr import ComparisonCExpr, IntegerCExpr, ListCExpr, VarCExpr
from Expander import expand, parse
from Reader import read
from Resolve import bindings
from Syntax import Integer, Scope, SList, Symbol


def counter() -> Callable[[], Scope]:
    # Every program under test mints its own uids from zero, so the scopes in
    # an expectation are a function of that program alone and never of the
    # process-wide counter `expand` draws from by default.
    uids = count()
    return lambda: Scope(next(uids))


# -- what counts as a condition: lambda-bound ----------------------------------


def test_a_reference_in_test_position_is_a_condition():
    expanded = expand(
        read("(lambda (x) (match x ((#t (prod#)) 1) ((#f (prod#)) 2)))"),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    assert conditions(Symbol("x", {Scope(0)}), program, table) == [
        VarCExpr(Symbol("x", {Scope(0)}))
    ]


def test_a_reference_in_a_branch_is_not_a_condition():
    expanded = expand(
        read("(lambda (x) (match #t ((#t (prod#)) x) ((#f (prod#)) 2)))"),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    assert conditions(Symbol("x", {Scope(0)}), program, table) == []


def test_the_whole_test_is_returned_not_the_occurrence():
    expanded = expand(
        read("(lambda (f x) (match (f x) ((#t (prod#)) 1) ((#f (prod#)) 2)))"),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    # Both parameters are binders of one form, so both carry its one scope.
    assert conditions(Symbol("x", {Scope(0)}), program, table) == [
        ListCExpr([Symbol("f", {Scope(0)}), Symbol("x", {Scope(0)})])
    ]


# -- what counts as a condition: letrec-bound ----------------------------------


def test_a_letrec_bound_reference_in_test_position_is_a_condition():
    expanded = expand(
        read("(letrec ((x 1)) (match x ((#t (prod#)) 2) ((#f (prod#)) 3)))"),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    assert conditions(Symbol("x", {Scope(0)}), program, table) == [
        VarCExpr(Symbol("x", {Scope(0)}))
    ]


def test_a_letrec_bound_reference_in_a_branch_is_not_a_condition():
    expanded = expand(
        read("(letrec ((x 1)) (match #t ((#t (prod#)) x) ((#f (prod#)) 2)))"),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    assert conditions(Symbol("x", {Scope(0)}), program, table) == []


def test_a_letrec_bound_application_is_returned_whole():
    expanded = expand(
        read(
            "(letrec ((f (lambda (y) y)) (x 1))"
            " (match (f x) ((#t (prod#)) 2) ((#f (prod#)) 3)))"
        ),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    assert conditions(Symbol("x", {Scope(0)}), program, table) == [
        ListCExpr([Symbol("f", {Scope(0)}), Symbol("x", {Scope(0)})])
    ]


def test_a_letrec_initializer_and_body_are_both_regions():
    expanded = expand(
        read(
            "(letrec ((f (match f ((#t (prod#)) 1) ((#f (prod#)) 2))))"
            " (match (f 3) ((#t (prod#)) 4) ((#f (prod#)) 5)))"
        ),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    # f is visible in its own initializer, so one variable reaches both
    # regions; the initializer's test comes before the body's.
    assert conditions(Symbol("f", {Scope(0)}), program, table) == [
        VarCExpr(Symbol("f", {Scope(0)})),
        ListCExpr([Symbol("f", {Scope(0)}), Integer(3)]),
    ]


def test_a_letrec_region_spans_sibling_initializers():
    expanded = expand(
        read(
            "(letrec ((f (match g ((#t (prod#)) 1) ((#f (prod#)) 2)))"
            "         (g (match f ((#t (prod#)) 3) ((#f (prod#)) 4))))"
            " 5)"
        ),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    # Each name is visible from the other's initializer, which is what makes
    # letrec bindings mutually recursive.
    assert conditions(Symbol("f", {Scope(0)}), program, table) == [
        VarCExpr(Symbol("f", {Scope(0)}))
    ]
    assert conditions(Symbol("g", {Scope(0)}), program, table) == [
        VarCExpr(Symbol("g", {Scope(0)}))
    ]


# -- variable identity across regions and shadows ------------------------------


def test_a_reference_from_an_enclosing_lambda_is_found():
    expanded = expand(
        read("(lambda (x) (lambda (y) (match x ((#t (prod#)) 1) ((#f (prod#)) 2))))"),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    assert conditions(Symbol("x", {Scope(0)}), program, table) == [
        VarCExpr(Symbol("x", {Scope(0)}))
    ]


def test_a_reference_from_an_enclosing_letrec_is_found():
    expanded = expand(
        read(
            "(letrec ((x 1)) (lambda (y) (match x ((#t (prod#)) 2) ((#f (prod#)) 3))))"
        ),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    assert conditions(Symbol("x", {Scope(0)}), program, table) == [
        VarCExpr(Symbol("x", {Scope(0)}))
    ]


def test_an_adjacent_lambda_shadow_belongs_to_the_inner_binder():
    expanded = expand(
        read("(lambda (x) (lambda (x) (match x ((#t (prod#)) 1) ((#f (prod#)) 2))))"),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    # The occurrence resolves to the inner binder, so the outer variable has
    # no condition even though the identifier matches.
    assert conditions(Symbol("x", {Scope(0)}), program, table) == []
    assert conditions(Symbol("x", {Scope(0), Scope(1)}), program, table) == [
        VarCExpr(Symbol("x", {Scope(0), Scope(1)}))
    ]


def test_a_three_deep_lambda_shadow_belongs_to_the_innermost_binder():
    expanded = expand(
        read(
            "(lambda (x) (lambda (x) (lambda (x)"
            " (match x ((#t (prod#)) 1) ((#f (prod#)) 2)))))"
        ),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    assert conditions(Symbol("x", {Scope(0)}), program, table) == []
    assert conditions(Symbol("x", {Scope(0), Scope(1)}), program, table) == []
    assert conditions(Symbol("x", {Scope(0), Scope(1), Scope(2)}), program, table) == [
        VarCExpr(Symbol("x", {Scope(0), Scope(1), Scope(2)}))
    ]


def test_an_interleaved_shadow_belongs_to_its_own_binder():
    expanded = expand(
        read(
            "(lambda (x) (lambda (y) (lambda (x)"
            " (match y ((#t (prod#)) 1) ((#f (prod#)) 2)))))"
        ),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    # The reference is to y, so neither x binder claims the condition — the
    # innermost binding form is not automatically the answer.
    assert conditions(Symbol("y", {Scope(0), Scope(1)}), program, table) == [
        VarCExpr(Symbol("y", {Scope(0), Scope(1)}))
    ]
    assert conditions(Symbol("x", {Scope(0)}), program, table) == []
    assert conditions(Symbol("x", {Scope(0), Scope(1), Scope(2)}), program, table) == []


def test_an_adjacent_letrec_shadow_belongs_to_the_inner_binder():
    expanded = expand(
        read(
            "(letrec ((x 1)) (letrec ((x 2))"
            " (match x ((#t (prod#)) 3) ((#f (prod#)) 4))))"
        ),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    assert conditions(Symbol("x", {Scope(0)}), program, table) == []
    assert conditions(Symbol("x", {Scope(0), Scope(1)}), program, table) == [
        VarCExpr(Symbol("x", {Scope(0), Scope(1)}))
    ]


def test_a_letrec_name_shadows_a_lambda_parameter():
    expanded = expand(
        read(
            "(lambda (x) (letrec ((x 1)) (match x ((#t (prod#)) 2) ((#f (prod#)) 3))))"
        ),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    assert conditions(Symbol("x", {Scope(0)}), program, table) == []
    assert conditions(Symbol("x", {Scope(0), Scope(1)}), program, table) == [
        VarCExpr(Symbol("x", {Scope(0), Scope(1)}))
    ]


def test_a_lambda_parameter_shadows_a_letrec_name():
    expanded = expand(
        read(
            "(letrec ((x 1)) (lambda (x) (match x ((#t (prod#)) 2) ((#f (prod#)) 3))))"
        ),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    assert conditions(Symbol("x", {Scope(0)}), program, table) == []
    assert conditions(Symbol("x", {Scope(0), Scope(1)}), program, table) == [
        VarCExpr(Symbol("x", {Scope(0), Scope(1)}))
    ]


# -- traversal -----------------------------------------------------------------


def test_a_nested_match_in_a_scrutinee_has_a_condition_of_its_own():
    expanded = expand(
        read(
            "(lambda (x)"
            " (match (match x ((#t (prod#)) 1) ((#f (prod#)) 2))"
            "   ((#t (prod#)) 3)"
            "   ((#f (prod#)) 4)))"
        ),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    x = Symbol("x", {Scope(0)})
    inner = [
        Symbol("match"),
        x,
        SList([SList([Symbol("#t"), SList([Symbol("prod#")])]), Integer(1)]),
        SList([SList([Symbol("#f"), SList([Symbol("prod#")])]), Integer(2)]),
    ]

    # The outer scrutinee is the whole inner `match`, which the condition
    # language has no constructor for and reads as the opaque list it is; the
    # inner scrutinee is the variable.
    assert conditions(x, program, table) == [ListCExpr(inner), VarCExpr(x)]


def test_sibling_conditions_come_out_in_source_order():
    expanded = expand(
        read(
            "(lambda (f g h x)"
            " (f (match (g x) ((#t (prod#)) 1) ((#f (prod#)) 2))"
            "    (match (h x) ((#t (prod#)) 3) ((#f (prod#)) 4))))"
        ),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    assert conditions(Symbol("x", {Scope(0)}), program, table) == [
        ListCExpr([Symbol("g", {Scope(0)}), Symbol("x", {Scope(0)})]),
        ListCExpr([Symbol("h", {Scope(0)}), Symbol("x", {Scope(0)})]),
    ]


def test_equal_tests_are_reported_once_per_if():
    expanded = expand(
        read(
            "(lambda (f x)"
            " (f (match x ((#t (prod#)) 1) ((#f (prod#)) 2))"
            "    (match x ((#t (prod#)) 3) ((#f (prod#)) 4))))"
        ),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    assert conditions(Symbol("x", {Scope(0)}), program, table) == [
        VarCExpr(Symbol("x", {Scope(0)})),
        VarCExpr(Symbol("x", {Scope(0)})),
    ]


def test_a_lambda_parameter_is_not_a_use():
    expanded = expand(
        read("(lambda (x) (match (lambda (y) 1) ((#t (prod#)) 1) ((#f (prod#)) 2)))"),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    # y occurs in the test, but only as a binding position.
    assert conditions(Symbol("y", {Scope(0), Scope(1)}), program, table) == []


def test_a_letrec_name_is_not_a_use():
    expanded = expand(
        read("(match (letrec ((g 1)) 2) ((#t (prod#)) 1) ((#f (prod#)) 2))"),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    # g occurs in the test, but only as a binding position.
    assert conditions(Symbol("g", {Scope(0)}), program, table) == []


def test_a_quoted_occurrence_is_not_the_bound_variable():
    expanded = expand(
        read(
            "(lambda (x)"
            " (match #t"
            "   ((#t (prod#)) (quote (match x ((#t (prod#)) 1) ((#f (prod#)) 2))))"
            "   ((#f (prod#)) 2)))"
        ),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    # The scope pass skips quoted data, so the quoted x carries no scopes and
    # is not the variable the parameter binds.
    assert conditions(Symbol("x", {Scope(0)}), program, table) == []


def test_a_program_with_no_if_has_no_conditions():
    expanded = expand(read("(lambda (f x) (f x))"), fresh=counter())
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    assert conditions(Symbol("x", {Scope(0)}), program, table) == []


def test_a_built_in_test_is_lifted_to_its_own_form():
    expanded = expand(
        read("(lambda (x) (match (lt x 1) ((#t (prod#)) 2) ((#f (prod#)) 3)))"),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    # The test is not reported as an application: `conditions` hands back the
    # typed form, which is what the analysis pattern-matches on.
    assert conditions(Symbol("x", {Scope(0)}), program, table) == [
        ComparisonCExpr(VarCExpr(Symbol("x", {Scope(0)})), IntegerCExpr(Integer(1)))
    ]


# -- ascribed binding positions --------------------------------------------------


def test_an_ascribed_letrec_name_is_not_a_use():
    # The ascribed twin of `test_a_letrec_name_is_not_a_use`: wrapping the
    # binder must not turn its binding position into an occurrence.
    expanded = expand(
        read(
            "(match (letrec (((ascribe# g @int) 1)) 2)"
            " ((#t (prod#)) 1) ((#f (prod#)) 2))"
        ),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    assert conditions(Symbol("g", {Scope(0)}), program, table) == []


def test_an_ascribed_letrec_initializer_is_still_searched():
    # The fix must not lose the initializer while dropping the binder: `x` is
    # used inside the initializer of an ascribed binding.
    expanded = expand(
        read(
            "(lambda (x)"
            " (match (letrec (((ascribe# g @int) x)) 2)"
            "  ((#t (prod#)) 1) ((#f (prod#)) 2)))"
        ),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    assert conditions(Symbol("x", {Scope(0)}), program, table) == [
        ListCExpr(
            [
                Symbol("letrec"),
                SList(
                    [
                        SList(
                            [
                                SList(
                                    [
                                        Symbol("ascribe#"),
                                        Symbol("g", {Scope(0), Scope(1)}),
                                        Symbol("@int"),
                                    ]
                                ),
                                Symbol("x", {Scope(0)}),
                            ]
                        )
                    ]
                ),
                Integer(2),
            ]
        )
    ]


def test_an_ascribed_type_operand_is_not_a_use():
    # `@int` is a type spelling, not a program variable: it must not read as
    # an occurrence of a free symbol that happens to share its spelling.
    expanded = expand(
        read(
            "(match (letrec (((ascribe# g @int) 1)) 2)"
            " ((#t (prod#)) 1) ((#f (prod#)) 2))"
        ),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    assert conditions(Symbol("@int"), program, table) == []
    assert conditions(Symbol("ascribe#"), program, table) == []


def test_a_define_name_is_not_a_use():
    expanded = expand(
        read("(define g (match g ((#t (prod#)) 1) ((#f (prod#)) 2)))"), fresh=counter()
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    # The occurrence in the test counts; the binding position does not.
    assert conditions(Symbol("g"), program, table) == [VarCExpr(Symbol("g"))]


def test_an_ascribed_define_name_is_not_a_use():
    expanded = expand(
        read("(define (ascribe# g @int) (match 1 ((#t (prod#)) 2) ((#f (prod#)) 3)))"),
        fresh=counter(),
    )
    program = parse(expanded, bindings(expanded))
    table = bindings(program)

    assert conditions(Symbol("g"), program, table) == []
    assert conditions(Symbol("@int"), program, table) == []
