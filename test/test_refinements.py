from collections.abc import Callable
from itertools import count, product

from hypothesis import given
from hypothesis import strategies as st

from Analysis import GROUNDS, disjoint, refinements, unreachable
from Analysis.Path import Path
from Analysis.Support import Support
from Evaluate.Builtins import eq
from Expander import expand, parse
from Reader import read
from Resolve import bindings
from Syntax import Boolean, Char, Expr, Integer, Scope, Symbol, Value
from Syntax.Type import (
    BoolType,
    CharType,
    Context,
    FunType,
    IntType,
    ListType,
    ProdType,
    Type,
    VariadicType,
)

LOWER, UPPER = Integer.bounds()


def counter() -> Callable[[], Scope]:
    uids = count()
    return lambda: Scope(next(uids))


def program_of(source: str):
    expanded = expand(read(source), fresh=counter())
    program = parse(expanded, bindings(expanded))
    return program, bindings(program)


def var(identifier: str, *uids: int) -> Symbol:
    return Symbol(identifier, {Scope(uid) for uid in uids})


def context(ascriptions: dict[Symbol, Type] | None = None) -> Context:
    # The ascriptions the walk reads, asserted rather than checked: these
    # programs are given to the walk directly and never to `check`, so nothing
    # has put a binder's type in the store for it. A fresh context per test,
    # because `register` mutates.
    rets: Context = Context.default()

    for binder, ty in (ascriptions or {}).items():
        rets.register(binder, ty)

    return rets


# -- refinements along a path ----------------------------------------------------


def test_a_consequent_use_keeps_the_values_below_the_bound():
    program, table = program_of("(lambda (x) (if (lt x 5) x 0))")
    x = var("x", 0)

    # The series opens on the ground range of the type and ends on the support
    # at the occurrence; here one guard contributes one refinement.
    assert refinements(x, Path([2, 2]), program, table, context()) == [
        Support.integers(),
        Support.of([(LOWER, 4)]),
    ]


def test_an_alternative_use_keeps_the_complement():
    program, table = program_of("(lambda (x) (if (lt x 5) 0 x))")
    x = var("x", 0)

    assert refinements(x, Path([2, 3]), program, table, context()) == [
        Support.integers(),
        Support.of([(5, UPPER)]),
    ]


def test_a_failed_equality_punches_a_hole():
    program, table = program_of("(lambda (x) (if (eq x 1) 0 x))")
    x = var("x", 0)

    # The alternative knows exactly one value is impossible, so the support is
    # the ground range with a single point removed — two intervals.
    assert refinements(x, Path([2, 3]), program, table, context()) == [
        Support.integers(),
        Support.of([(LOWER, 0), (2, UPPER)]),
    ]


def test_nested_guards_tighten_one_per_step():
    program, table = program_of("(lambda (x) (if (lt x 10) (if (lt 5 x) x 0) 0))")
    x = var("x", 0)

    assert refinements(x, Path([2, 2, 2]), program, table, context()) == [
        Support.integers(),
        Support.of([(LOWER, 9)]),
        Support.of([(6, 9)]),
    ]


def test_negation_flips_the_polarity_of_its_operand():
    program, table = program_of("(lambda (x) (if (not (lt x 5)) x 0))")
    x = var("x", 0)

    assert refinements(x, Path([2, 2]), program, table, context()) == [
        Support.integers(),
        Support.of([(5, UPPER)]),
    ]


def test_a_conjunction_intersects_and_its_failure_unions():
    program, table = program_of("(lambda (x) (if (and (lt 0 x) (lt x 10)) x x))")
    x = var("x", 0)

    # Passing the conjunction pins both conjuncts.
    assert refinements(x, Path([2, 2]), program, table, context())[-1] == Support.of(
        [(1, 9)]
    )

    # Failing it promises only that one conjunct failed, so the two negations
    # union (De Morgan) and the support splits around the window.
    assert refinements(x, Path([2, 3]), program, table, context())[-1] == Support.of(
        [(LOWER, 0), (10, UPPER)]
    )


def test_a_failed_disjunction_intersects_the_negations():
    program, table = program_of("(lambda (x) (if (or (lt x 0) (lt 10 x)) 0 x))")
    x = var("x", 0)

    assert refinements(x, Path([2, 3]), program, table, context())[-1] == Support.of(
        [(0, 10)]
    )


def test_le_is_read_as_the_disjunction_of_lt_and_eq():
    # `le` has no condition constructor, so its guards arrive as applications;
    # the walk rewrites `(le a b)` to `(or (lt a b) (eq a b))` and reads that.
    program, table = program_of("(lambda (x) (if (le x 5) x 0))")
    series = refinements(var("x", 0), Path([2, 2]), program, table, context())
    assert series[-1] == Support.of([(LOWER, 5)])

    # The mirrored order rewrites too: failing `5 <= x` is exactly `x < 5`.
    program, table = program_of("(lambda (x) (if (le 5 x) 0 x))")
    series = refinements(var("x", 0), Path([2, 3]), program, table, context())
    assert series[-1] == Support.of([(LOWER, 4)])


def test_a_shadowed_relation_is_an_application_and_does_not_refine():
    # This `le` is a parameter, not the built-in, so the guard says nothing a
    # sound reading can use.
    program, table = program_of("(lambda (le x) (if (le x 5) x 0))")
    x = var("x", 0)

    assert refinements(x, Path([2, 2]), program, table, context()) == [
        Support.integers(),
        Support.integers(),
    ]


def test_a_guard_over_another_variable_does_not_refine():
    program, table = program_of("(lambda (x y) (if (lt y 5) x 0))")
    x = var("x", 0)

    assert refinements(x, Path([2, 2]), program, table, context()) == [
        Support.integers(),
        Support.integers(),
    ]


def test_an_opaque_test_does_not_refine():
    # `(mod x 3)` is an application inside the equality, so the walk cannot
    # read the test as a bound on `x` and must not pretend to.
    program, table = program_of("(lambda (x) (if (eq (mod x 3) 0) x 0))")
    x = var("x", 0)

    assert refinements(x, Path([2, 2]), program, table, context()) == [
        Support.integers(),
        Support.integers(),
    ]


def test_contradictory_guards_empty_the_support():
    program, table = program_of("(lambda (x) (if (lt x 5) (if (lt 9 x) x 0) 0))")
    x = var("x", 0)

    series = refinements(x, Path([2, 2, 2]), program, table, context())
    assert series == [
        Support.integers(),
        Support.of([(LOWER, 4)]),
        Support.empty(),
    ]
    assert series[-1].is_empty()


def test_a_boolean_test_is_a_symbol_and_does_not_refine():
    # `#f` reads as a symbol, not a literal: the walk sees a variable it knows
    # nothing about and must not read any truth into it.
    program, table = program_of("(lambda (x) (if #f x 0))")
    x = var("x", 0)

    assert refinements(x, Path([2, 2]), program, table, context()) == [
        Support.integers(),
        Support.integers(),
    ]


# -- type-disjoint guards --------------------------------------------------------


def test_an_integer_compared_to_a_boolean_empties_the_consequent():
    # `#t` reads as an ordinary symbol, so the ascription the default context
    # gives it is the only thing that makes this guard a type mismatch.
    program, table = program_of("(lambda (x) (if (eq x #t) x 0))")
    x = var("x", 0)
    ascribed = context({x: IntType()})

    assert refinements(x, Path([2, 2]), program, table, ascribed) == [
        Support.integers(),
        Support.empty(),
    ]

    # Failing the guard is what every state does, which refines nothing.
    assert refinements(x, Path([2, 3]), program, table, ascribed) == [
        Support.integers(),
        Support.integers(),
    ]


def test_a_character_never_equals_an_integer():
    program, table = program_of(r"(lambda (x) (if (eq x #\a) x 0))")
    x = var("x", 0)

    series = refinements(x, Path([2, 2]), program, table, context({x: IntType()}))
    assert series[-1].is_empty()


def test_an_arithmetic_result_compared_to_a_boolean_empties_the_consequent():
    # The operand is an application, typed by the return of the ascription its
    # head carries in the default context.
    program, table = program_of("(lambda (x) (if (eq (+ x 1) #t) x 0))")
    x = var("x", 0)

    series = refinements(x, Path([2, 2]), program, table, context({x: IntType()}))
    assert series[-1].is_empty()


def test_a_comparison_compared_to_an_integer_empties_the_consequent():
    # No ascription is needed at all here: a comparison answers with a boolean
    # whatever it compares.
    program, table = program_of("(lambda (x) (if (eq (lt x 0) 1) x 0))")
    x = var("x", 0)

    series = refinements(x, Path([2, 2]), program, table, context())
    assert series[-1].is_empty()


def test_operands_of_the_same_type_still_refine_by_value():
    # The disjointness reading sits in front of the point reading, so a guard
    # it declines must still arrive there.
    program, table = program_of("(lambda (x) (if (eq x 1) x 0))")
    x = var("x", 0)

    assert refinements(x, Path([2, 2]), program, table, context({x: IntType()})) == [
        Support.integers(),
        Support.point(1),
    ]


def test_an_unascribed_operand_eliminates_nothing():
    program, table = program_of("(lambda (x y) (if (eq x y) x 0))")
    x = var("x", 0)

    series = refinements(x, Path([2, 2]), program, table, context({x: IntType()}))
    assert series[-1] == Support.integers()


def test_a_context_that_ascribes_nothing_eliminates_nothing():
    program, table = program_of("(lambda (x) (if (eq x #t) x 0))")
    x = var("x", 0)

    assert refinements(x, Path([2, 2]), program, table, context()) == [
        Support.integers(),
        Support.integers(),
    ]


def test_a_disjoint_equality_under_a_negation_is_always_true():
    program, table = program_of("(lambda (x) (if (not (eq x #t)) x 0))")
    x = var("x", 0)
    ascribed = context({x: IntType()})

    assert refinements(x, Path([2, 2]), program, table, ascribed)[-1] == (
        Support.integers()
    )
    assert refinements(x, Path([2, 3]), program, table, ascribed)[-1].is_empty()


def test_a_disjoint_disjunct_drops_out_of_the_disjunction():
    # `(or #f r)` is `r`, so the live disjunct is the whole reading.
    program, table = program_of("(lambda (x) (if (or (eq x #t) (lt x 5)) x 0))")
    x = var("x", 0)

    series = refinements(x, Path([2, 2]), program, table, context({x: IntType()}))
    assert series[-1] == Support.of([(LOWER, 4)])


def test_a_disjoint_conjunct_empties_the_conjunction():
    program, table = program_of("(lambda (x) (if (and (eq x #t) (lt x 5)) x 0))")
    x = var("x", 0)
    ascribed = context({x: IntType()})

    assert refinements(x, Path([2, 2]), program, table, ascribed)[-1].is_empty()
    assert refinements(x, Path([2, 3]), program, table, ascribed)[-1] == (
        Support.integers()
    )


def test_a_disjoint_equality_under_le_says_nothing():
    # `(le x #t)` rewrites to `(or (lt x #t) (eq x #t))`, whose `eq` half is
    # statically false — but `le` *raises* on a boolean rather than answering
    # `#f`, so neither branch may be called dead. The opaque `lt` half is what
    # keeps the union honest.
    program, table = program_of("(lambda (x) (if (le x #t) x 0))")
    x = var("x", 0)
    ascribed = context({x: IntType()})

    assert refinements(x, Path([2, 2]), program, table, ascribed)[-1] == (
        Support.integers()
    )
    assert refinements(x, Path([2, 3]), program, table, ascribed)[-1] == (
        Support.integers()
    )


def test_a_shadowed_eq_is_an_application_and_eliminates_nothing():
    # This `eq` is a parameter, not the built-in, so the guard converts to an
    # application and no reading of it is available at all.
    program, table = program_of("(lambda (eq x) (if (eq x #t) x 0))")
    x = var("x", 0)

    series = refinements(x, Path([2, 2]), program, table, context({x: IntType()}))
    assert series[-1] == Support.integers()


def test_a_constructor_spelling_is_not_read_through_an_ascription():
    # `list` heads a construction, whose type is the one being checked against
    # and never the return of something ascribed to the spelling. Reading the
    # ascription instead would call this branch dead on the strength of an
    # ascription that does not describe the operand.
    program, table = program_of("(lambda (x) (if (eq (list x) #t) x 0))")
    x = var("x", 0)
    ascribed = context(
        {
            x: IntType(),
            Symbol("list"): FunType(
                return_type=IntType(),
                argument_type=ProdType({Symbol("a"): IntType()}),
            ),
        }
    )

    series = refinements(x, Path([2, 2]), program, table, ascribed)
    assert series[-1] == Support.integers()


# -- unreachable branches --------------------------------------------------------


def test_a_type_mismatched_guard_makes_its_consequent_unreachable():
    program, table = program_of("(lambda (x) (if (eq x #t) x 0))")
    ascribed = context({var("x", 0): IntType()})

    assert unreachable(Path([2, 2]), program, table, ascribed)
    assert not unreachable(Path([2, 3]), program, table, ascribed)


def test_a_bound_on_one_variable_does_not_make_a_branch_unreachable():
    # Reachability is asked of the path and not of a variable, so a guard that
    # only bounds one — even contradictorily — is not its business. That
    # question is `refinements`'.
    program, table = program_of("(lambda (x) (if (lt x 5) (if (lt 9 x) x 0) 0))")

    assert not unreachable(Path([2, 2, 2]), program, table, context())
    assert refinements(var("x", 0), Path([2, 2, 2]), program, table, context())[
        -1
    ].is_empty()


# -- the disjointness relation ---------------------------------------------------


def test_the_ground_types_are_pairwise_disjoint_and_symmetrically_so():
    grounds = [IntType(), CharType(), BoolType()]

    for left, right in product(grounds, repeat=2):
        assert disjoint(left, right) == (left != right)
        assert disjoint(left, right) == disjoint(right, left)


def test_two_list_types_are_not_disjoint():
    # The empty list inhabits `(@list t)` for every `t` and compares equal to
    # itself whatever the element types say.
    assert not disjoint(ListType(IntType()), ListType(BoolType()))


def test_a_function_type_is_never_disjoint():
    # `eq` raises on a procedure instead of answering `#f`, so a function type
    # must never be what makes a branch dead: that would delete an error.
    procedure = FunType(return_type=IntType(), argument_type=ProdType())

    assert not disjoint(IntType(), procedure)
    assert not disjoint(procedure, BoolType())


def test_a_variadic_type_is_never_disjoint():
    # A variadic type classifies no term at all, so calling it disjoint from
    # anything would be vacuously true.
    assert not disjoint(IntType(), VariadicType(IntType()))


# The values each ground type has at run time. `@bool` has two
# representations: the symbols `#t` and `#f` that source text reads as, and the
# `Boolean` nodes a comparison answers with.
INHABITANTS: dict[Type, list[Expr]] = {
    IntType(): [Integer(0), Integer(1)],
    CharType(): [Char(0), Char(ord("a"))],
    BoolType(): [Boolean(True), Boolean(False), Symbol("#t"), Symbol("#f")],
}


def test_disjoint_types_share_no_value_the_evaluator_calls_equal():
    # The relation is a claim about `eq`, so `eq` is what checks it: for every
    # disjoint pair, every pair of inhabitants must *answer*, and answer false.
    # A ground type added without inhabitants would go unchecked, so the two
    # listings are pinned to each other.
    assert set(INHABITANTS) == set(GROUNDS)

    for left, right in product(INHABITANTS, repeat=2):
        if not disjoint(left, right):
            continue

        for value, other in product(INHABITANTS[left], INHABITANTS[right]):
            arguments: list[Value] = [value, other]

            assert eq(arguments) == Boolean(False)


# -- the support domain ----------------------------------------------------------


def test_construction_merges_overlapping_and_abutting_intervals():
    assert Support.of([(0, 3), (4, 6)]) == Support.of([(0, 6)])
    assert Support.of([(0, 5), (3, 8)]) == Support.of([(0, 8)])
    assert Support.of([(3, 4), (0, 1)]).intervals == ((0, 1), (3, 4))


def test_constructors_clip_to_the_ground_range():
    assert Support.less_than(UPPER + 2) == Support.integers()
    assert Support.less_than(LOWER) == Support.empty()
    assert Support.point(5) == Support.of([(5, 5)])
    assert Support.point(5).cardinality() == 1


def test_membership_follows_the_intervals():
    support = Support.of([(0, 2), (10, 12)])

    assert 1 in support
    assert 11 in support
    assert 5 not in support


# A small window keeps the examples legible; every operation is closed over the
# full ground range regardless.
intervals = st.lists(
    st.tuples(
        st.integers(min_value=-50, max_value=50),
        st.integers(min_value=-50, max_value=50),
    ).map(lambda pair: (min(pair), max(pair))),
    max_size=4,
)
supports = intervals.map(Support.of)


@given(supports)
def test_complement_is_an_involution(support: Support):
    assert support.complement().complement() == support


@given(supports)
def test_a_support_and_its_complement_partition_the_ground_range(support: Support):
    assert support.intersection(support.complement()).is_empty()
    assert support.union(support.complement()) == Support.integers()
    assert (
        support.cardinality() + support.complement().cardinality()
        == Support.integers().cardinality()
    )


@given(supports, supports)
def test_cardinality_obeys_inclusion_exclusion(ours: Support, theirs: Support):
    assert (
        ours.union(theirs).cardinality() + ours.intersection(theirs).cardinality()
        == ours.cardinality() + theirs.cardinality()
    )


@given(supports, supports)
def test_complement_distributes_by_de_morgan(ours: Support, theirs: Support):
    assert ours.union(theirs).complement() == ours.complement().intersection(
        theirs.complement()
    )
