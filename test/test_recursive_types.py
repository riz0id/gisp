from collections.abc import Callable
from itertools import count
from typing import Final

import pytest

from Analysis import disjoint
from Expander import expand, parse
from Reader import read
from Resolve import BindingTable, bindings
from Syntax import Expr, Scope, Symbol
from Syntax.Type import (
    BOOL,
    Context,
    FunType,
    IntType,
    MuType,
    ProdType,
    SumType,
    Type,
    VariadicType,
    VarType,
    constructor_arity,
    constructors,
    is_constructor,
    is_nullary_type,
    list_of,
    type_constructor_arity,
    type_constructor_symbol,
    unfold,
)
from TypeCheck import TypeChecker, TypeCheckError

# Fixtures inject deterministic counters from the 1600s so no expectation
# depends on the global fresh-uid counter and no uid collides with another test
# file's hand-written scopes.

# The one type variable every fixture binds. A recursive type binds its own
# variable, so nothing here needs a fresh one.
A: Final[Symbol] = Symbol("a")
B: Final[Symbol] = Symbol("b")


def counter(start: int) -> Callable[[], Scope]:
    uids = count(start)
    return lambda: Scope(next(uids))


def program_of(source: str, expand_at: int) -> tuple[Expr, BindingTable]:
    expanded = expand(read(source), fresh=counter(expand_at))
    program = parse(expanded, bindings(expanded))
    return program, bindings(program)


def nat() -> MuType:
    # (@mu a (@sum (zero (@prod)) (succ a)))
    return MuType(
        A,
        SumType(
            {
                Symbol("zero"): ProdType(),
                Symbol("succ"): VarType(A),
            }
        ),
    )


def ints() -> MuType:
    # (@mu a (@sum (nil (@prod)) (cons (@prod (head @int) (tail a)))))
    return MuType(
        A,
        SumType(
            {
                Symbol("nil"): ProdType(),
                Symbol("cons"): ProdType(
                    {
                        Symbol("head"): IntType(),
                        Symbol("tail"): VarType(A),
                    }
                ),
            }
        ),
    )


# -- unfolding ---------------------------------------------------------------------


def test_unfolding_replaces_the_variable_with_the_whole_type():
    assert unfold(nat()) == SumType(
        {
            Symbol("zero"): ProdType(),
            Symbol("succ"): nat(),
        }
    )


def test_unfolding_leaves_the_recursive_positions_folded():
    # The property the whole design rests on: only the top node changes form,
    # so nothing below it is ever compared in a form other than the one it was
    # written in, and no recursive equality is needed.
    unfolded = unfold(nat())

    assert isinstance(unfolded, SumType)
    assert unfolded.cons[Symbol("succ")] == nat()


def test_unfolding_descends_through_every_sort_that_has_parts():
    element = MuType(A, list_of(VarType(A)))
    assert unfold(element) == list_of(element)

    arguments = MuType(
        A,
        FunType(
            return_type=IntType(),
            argument_type=ProdType({Symbol("f"): VarType(A)}),
        ),
    )
    assert unfold(arguments) == FunType(
        return_type=IntType(),
        argument_type=ProdType({Symbol("f"): arguments}),
    )

    run = MuType(
        A,
        FunType(return_type=IntType(), argument_type=VariadicType(VarType(A))),
    )
    assert unfold(run) == FunType(
        return_type=IntType(), argument_type=VariadicType(run)
    )


def test_unfolding_leaves_a_rebound_variable_alone():
    # The inner `@mu` binds the same variable, so every occurrence beneath it
    # is an occurrence of that binder and not of the one being replaced.
    inner = MuType(A, ProdType({Symbol("y"): VarType(A)}))
    outer = MuType(A, ProdType({Symbol("x"): inner}))
    unfolded = unfold(outer)

    assert isinstance(unfolded, ProdType)
    assert unfolded.names() == (Symbol("x"),)
    assert unfolded.types() == (inner,)


def test_unfolding_rebuilds_a_tuple_as_a_tuple():
    # `substitute` is the one place a product is reconstructed, and so the one
    # place a tuple could be collapsed into a record.
    recursive = MuType(A, ProdType([VarType(A)]))

    assert unfold(recursive) == ProdType([recursive])


# -- checking ----------------------------------------------------------------------


def test_a_recursive_sum_is_constructed_to_any_depth():
    for source in [
        "(zero (prod#))",
        "(succ (zero (prod#)))",
        "(succ (succ (succ (zero (prod#)))))",
    ]:
        program, table = program_of(source, 1600)
        TypeChecker(table=table).check(program, nat())


def test_a_payload_off_the_recursive_type_is_refused():
    program, table = program_of("(succ (#t (prod#)))", 1605)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, nat())


def test_a_constructor_the_recursive_type_does_not_have_is_refused():
    program, table = program_of("(pred (zero (prod#)))", 1606)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, nat())


def test_a_recursive_list_checks_fieldwise_at_every_depth():
    source = "(cons (prod# 1 (cons (prod# 2 (nil (prod#))))))"
    program, table = program_of(source, 1610)

    TypeChecker(table=table).check(program, ints())


def test_an_element_off_the_recursive_list_is_refused():
    program, table = program_of("(cons (prod# (#t (prod#)) (nil (prod#))))", 1611)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ints())


def test_a_recursive_list_is_not_a_list_of_the_other_element_type():
    program, table = program_of("(cons (prod# 1 (nil (prod#))))", 1612)

    TypeChecker(table=table).check(program, ints())
    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, nat())


# -- references --------------------------------------------------------------------


def test_a_reference_checks_at_the_recursive_type_it_is_ascribed():
    # The test that fails if unfolding is ever moved to the head of `check`:
    # the store holds `n` at the recursive type as written, and the body is
    # checked against that same type, so the comparison meets two folded types
    # and structural equality is the right relation.
    program, table = program_of("(lambda (n) n)", 1615)
    ty = FunType(return_type=nat(), argument_type=ProdType({Symbol("n"): nat()}))

    TypeChecker(table=table).check(program, ty)


def test_a_reference_off_the_recursive_type_is_refused():
    program, table = program_of("(lambda (n) n)", 1616)
    ty = FunType(return_type=ints(), argument_type=ProdType({Symbol("n"): nat()}))

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, ty)


def test_a_recursive_type_is_constructed_from_a_reference():
    program, table = program_of("(lambda (n) (succ n))", 1617)
    ty = FunType(return_type=nat(), argument_type=ProdType({Symbol("n"): nat()}))

    TypeChecker(table=table).check(program, ty)


# -- destructuring -----------------------------------------------------------------


def test_a_lambda_inhabits_a_recursive_function_type():
    # The only exercise of `abstraction`'s `@mu` arm: the `@fun` it needs is
    # visible only through the unfolding.
    program, table = program_of("(lambda (g) 1)", 1620)
    ty = MuType(
        A,
        FunType(
            return_type=IntType(),
            argument_type=ProdType({Symbol("g"): VarType(A)}),
        ),
    )

    TypeChecker(table=table).check(program, ty)


# -- declarations ------------------------------------------------------------------


def test_the_recursive_type_constructor_is_written_at_mu():
    store: Context = Context.default()

    assert type_constructor_symbol(nat(), store) == Symbol("@mu")
    assert type_constructor_arity(nat(), store) == 1


def test_a_variable_reports_the_type_it_is_bound_to():
    # A variable is not a constructor and has no arity of its own. It stands
    # for the type bound to it, so it answers whatever that type answers --
    # which is the whole of `(@mu a t)` for the variable the `@mu` binds.
    store: Context = Context.default().bind(A, nat())

    assert type_constructor_symbol(VarType(A), store) == Symbol("@mu")
    assert type_constructor_arity(VarType(A), store) == 1
    assert constructor_arity(VarType(A), store) == 1


def test_a_variable_bound_to_a_compound_type_reports_its_arity():
    # The reading that zero cannot express: a variable answers for the type
    # bound to it, whatever that type answers. `(@list t)` is the `@mu` over a
    # sum, so these are the `@mu` and `SumType` answers and not a sort of
    # their own — one type argument, and one payload per constructor.
    store: Context = Context.default().bind(A, list_of(IntType()))

    assert type_constructor_symbol(VarType(A), store) == Symbol("@mu")
    assert type_constructor_arity(VarType(A), store) == 1
    assert constructor_arity(VarType(A), store) == 1


def test_a_variable_is_constructed_however_its_binding_is():
    store: Context = Context.default().bind(A, list_of(IntType()))
    table: BindingTable = BindingTable.default()

    assert is_constructor(VarType(A), Symbol("cons"), store, table)
    assert is_constructor(VarType(A), Symbol("nil"), store, table)
    assert not is_constructor(VarType(A), Symbol("prod#"), store, table)
    # `list` and `.` construct nothing: the sum names its two constructors,
    # and neither of these is one of them.
    assert not is_constructor(VarType(A), Symbol("list"), store, table)
    assert not is_constructor(VarType(A), Symbol("."), store, table)


def test_a_variable_bound_to_a_ground_type_is_nullary():
    # The reading equality cannot make: `VarType(A)` is not one of the ground
    # instances, and yet it stands for one.
    store: Context = Context.default().bind(A, IntType())

    assert is_nullary_type(IntType(), store)
    assert is_nullary_type(VarType(A), store)
    assert not is_nullary_type(nat(), store)


def test_a_variable_bound_to_a_compound_type_is_not_nullary():
    store: Context = Context.default().bind(A, list_of(IntType()))

    assert not is_nullary_type(VarType(A), store)


@pytest.mark.parametrize(
    "query",
    [
        lambda ty, store: type_constructor_arity(ty, store),
        lambda ty, store: type_constructor_symbol(ty, store),
        lambda ty, store: constructor_arity(ty, store),
        lambda ty, store: is_constructor(ty, A, store, BindingTable.default()),
        lambda ty, store: is_nullary_type(ty, store),
    ],
)
def test_an_unbound_variable_is_refused(query: Callable[[Type, Context], object]):
    # Nothing binds `a`, so there is no type to answer for. Refused rather
    # than read as a nullary type, which is what zero would have claimed.
    with pytest.raises(ValueError):
        query(VarType(A), Context.default())


def test_an_inner_binding_shadows_an_outer_one():
    # `bind` extends functionally, so an inner `@mu` of the same variable
    # shadows the outer binding instead of conflicting with it the way
    # `register` would.
    store: Context = Context.default().bind(A, nat()).bind(A, list_of(IntType()))

    # Both bindings are a `@mu`, so the constructors are what tell them apart:
    # the variable reads through to the list's two and not to `nat`'s.
    assert constructors(VarType(A), store) is not None
    assert set(constructors(VarType(A), store) or {}) == {
        Symbol("cons"),
        Symbol("nil"),
    }


def test_the_constructors_are_read_under_the_binding_the_mu_makes():
    # The equivalence the `MuType` arms rest on: reading the body under the
    # binding the `@mu` makes answers what reading its unfolding answered.
    store: Context = Context.default()
    table: BindingTable = BindingTable.default()

    assert constructor_arity(nat(), store) == 1
    assert is_constructor(nat(), Symbol("succ"), store, table)
    assert is_constructor(nat(), Symbol("zero"), store, table)
    assert not is_constructor(nat(), Symbol("pred"), store, table)


# -- equality ----------------------------------------------------------------------


def test_a_recursive_type_is_structurally_equal_and_hashable():
    assert nat() == nat()
    assert hash(nat()) == hash(nat())
    assert nat() != ints()


def test_a_recursive_type_is_not_equal_to_its_unfolding():
    # Recorded rather than asserted as desirable: the two classify the same
    # terms, and making them one value is what a recursive type equality would
    # buy. Nothing in the checker reaches a comparison that needs it.
    assert nat() != unfold(nat())


def test_a_variable_is_equal_only_to_the_same_variable():
    assert VarType(A) == VarType(A)
    assert VarType(A) != VarType(B)
    assert VarType(A) != IntType()


# -- analysis ----------------------------------------------------------------------


def test_a_recursive_type_is_disjoint_from_nothing():
    # A recursive type is not a ground type, so the analysis stays
    # conservative and no branch is read as dead on account of one.
    assert not disjoint(nat(), IntType())
    assert not disjoint(BOOL, nat())
    assert not disjoint(nat(), ints())
