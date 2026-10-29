from collections.abc import Callable
from itertools import count

import pytest

from Analysis import (
    Guard,
    OccurrenceError,
    conditions,
    occurrences,
    path_conditions,
)
from Analysis.CExpr import ComparisonCExpr, IntegerCExpr, VarCExpr
from Analysis.Path import Path
from Expander import expand, parse
from Reader import read
from Resolve import bindings
from Syntax import Integer, Scope, Symbol


def counter() -> Callable[[], Scope]:
    uids = count()
    return lambda: Scope(next(uids))


def program_of(source: str):
    expanded = expand(read(source), fresh=counter())
    program = parse(expanded, bindings(expanded))
    return program, bindings(program)


def var(identifier: str, *uids: int) -> Symbol:
    return Symbol(identifier, {Scope(uid) for uid in uids})


def lt(left, right) -> ComparisonCExpr:
    return ComparisonCExpr(left, right)


# -- the two functions answer different questions ------------------------------


def test_sibling_ifs_give_one_list_but_four_distinct_occurrences():
    program, table = program_of(
        "(lambda (f x) (f (if (lt x 5) x 0) (if (lt 5 x) x 0)))"
    )
    x = var("x", 0)

    below = lt(VarCExpr(x), IntegerCExpr(Integer(5)))
    above = lt(IntegerCExpr(Integer(5)), VarCExpr(x))

    # Keyed on the variable: one list, both tests, for the whole program.
    assert conditions(x, program, table) == [below, above]

    # Keyed on the occurrence: four references, four different answers.
    assert occurrences(x, program, table) == [
        Path([2, 1, 1, 1]),
        Path([2, 1, 2]),
        Path([2, 2, 1, 2]),
        Path([2, 2, 2]),
    ]
    assert path_conditions(Path([2, 1, 1, 1]), program, table) == []
    assert path_conditions(Path([2, 1, 2]), program, table) == [
        Guard(test=below, consequent=True)
    ]
    assert path_conditions(Path([2, 2, 1, 2]), program, table) == []
    assert path_conditions(Path([2, 2, 2]), program, table) == [
        Guard(test=above, consequent=True)
    ]


def test_conjoining_the_flat_list_is_unsatisfiable_where_no_occurrence_is():
    program, table = program_of(
        "(lambda (f x) (f (if (lt x 5) x 0) (if (lt 5 x) x 0)))"
    )
    x = var("x", 0)

    below = lt(VarCExpr(x), IntegerCExpr(Integer(5)))
    above = lt(IntegerCExpr(Integer(5)), VarCExpr(x))

    # `x < 5` and `5 < x` both come back for `x`, so their conjunction is
    # empty: the region every use of `x` is reached in would be unreachable.
    harvested = conditions(x, program, table)
    assert below in harvested and above in harvested

    # No occurrence is subject to both, so no path condition is contradictory.
    for path_obj in occurrences(x, program, table):
        tests = [guard.test for guard in path_conditions(path_obj, program, table)]
        assert not (below in tests and above in tests)


def test_a_guard_that_does_not_mention_the_variable_is_still_a_guard():
    program, table = program_of("(lambda (x y) (if (lt y 5) x 0))")
    x = var("x", 0)

    # The test never names `x`, so the harvest reports nothing for it.
    assert conditions(x, program, table) == []

    # The `if` guards the use of `x` all the same, which is what the support of
    # `x` there depends on whenever `x` and `y` are dependent.
    assert occurrences(x, program, table) == [Path([2, 2])]
    assert path_conditions(Path([2, 2]), program, table) == [
        Guard(
            test=lt(VarCExpr(var("y", 0)), IntegerCExpr(Integer(5))),
            consequent=True,
        )
    ]


def test_an_occurrence_in_the_alternative_carries_the_opposite_polarity():
    program, table = program_of("(lambda (x) (if (lt x 5) 0 x))")
    x = var("x", 0)

    below = lt(VarCExpr(x), IntegerCExpr(Integer(5)))

    # The harvest hands back the test, which is the region the occurrence is
    # not in.
    assert conditions(x, program, table) == [below]

    assert path_conditions(Path([2, 3]), program, table) == [
        Guard(test=below, consequent=False)
    ]


def test_a_test_region_is_not_a_guard():
    program, table = program_of("(lambda (x) (if (if (lt x 5) 1 2) 3 x))")
    x = var("x", 0)

    harvested = conditions(x, program, table)

    # Both `if` forms mention `x`, so both tests are harvested: the outer test
    # whole, and the inner test it contains.
    assert len(harvested) == 2
    assert harvested[1] == lt(VarCExpr(x), IntegerCExpr(Integer(5)))

    # Only the outer form guards the occurrence. The inner one sits in a test
    # region: both of its branches flow into the outer test's value, so
    # reaching `x` commits to neither.
    assert path_conditions(Path([2, 3]), program, table) == [
        Guard(test=harvested[0], consequent=False)
    ]


def test_nested_ifs_tighten_along_the_path():
    program, table = program_of("(lambda (x) (if (lt x 10) (if (lt 5 x) x 0) 0))")
    x = var("x", 0)

    below = lt(VarCExpr(x), IntegerCExpr(Integer(10)))
    above = lt(IntegerCExpr(Integer(5)), VarCExpr(x))

    # Indistinguishable from the sibling case above: one flat list of two
    # tests, with nothing to say whether conjoining them is sound.
    assert conditions(x, program, table) == [below, above]

    # Here it is sound, and the walk says so by position.
    assert path_conditions(Path([2, 2, 2]), program, table) == [
        Guard(test=below, consequent=True),
        Guard(test=above, consequent=True),
    ]

    # The use inside the inner test is past the outer guard but not the inner.
    assert path_conditions(Path([2, 2, 1, 2]), program, table) == [
        Guard(test=below, consequent=True)
    ]


def test_a_letrec_initializer_is_walked_as_a_region():
    program, table = program_of("(letrec ((y 1) (x (if (lt y 5) y 0))) 2)")
    y = var("y", 0)

    # The binding position of `y` is not an occurrence; the two references in
    # the initializer are.
    assert occurrences(y, program, table) == [Path([1, 1, 1, 1, 1]), Path([1, 1, 1, 2])]
    assert path_conditions(Path([1, 1, 1, 2]), program, table) == [
        Guard(
            test=lt(VarCExpr(y), IntegerCExpr(Integer(5))),
            consequent=True,
        )
    ]
    assert path_conditions(Path([1, 1, 1, 1, 1]), program, table) == []


def test_every_occurrence_has_a_path_the_walk_accepts():
    program, table = program_of(
        "(letrec ((f (lambda (x) (if (lt x 5) x (quote x))))) (if (lt 1 2) (f 3) 4))"
    )

    for identifier in ("f", "x"):
        for uid in (0, 1):
            for path in occurrences(var(identifier, uid), program, table):
                path_conditions(path, program, table)


# -- what is not an occurrence -------------------------------------------------


def test_a_parameter_list_is_not_an_occurrence():
    program, table = program_of("(lambda (x) x)")

    with pytest.raises(OccurrenceError):
        path_conditions(Path([1, 0]), program, table)


def test_a_letrec_name_is_not_an_occurrence():
    program, table = program_of("(letrec ((x 1)) x)")

    with pytest.raises(OccurrenceError):
        path_conditions(Path([1, 0, 0]), program, table)

    # The initializer beside it is an ordinary region.
    assert path_conditions(Path([1, 0, 1]), program, table) == []


def test_quoted_data_is_not_an_occurrence():
    program, table = program_of("(lambda (x) (quote (if x 1 2)))")

    with pytest.raises(OccurrenceError):
        path_conditions(Path([2, 1]), program, table)


def test_a_path_off_the_tree_is_refused():
    program, table = program_of("(lambda (x) (if (lt x 5) x 0))")

    with pytest.raises(OccurrenceError):
        path_conditions(Path([2, 9]), program, table)

    with pytest.raises(OccurrenceError):
        path_conditions(Path([2, 3, 0]), program, table)


# -- ascribed binding positions --------------------------------------------------


def test_an_ascribed_letrec_name_is_not_an_occurrence():
    # The ascribed twin of `test_a_letrec_name_is_not_an_occurrence`. Position
    # 0 of a pair is a binding site whatever shape the binder has, so the
    # refusal must survive the wrapper.
    program, table = program_of("(letrec (((ascribe# x @int) 1)) x)")

    with pytest.raises(OccurrenceError):
        path_conditions(Path([1, 0, 0]), program, table)

    # The initializer beside it is an ordinary region.
    assert path_conditions(Path([1, 0, 1]), program, table) == []


def test_no_path_reaches_inside_an_ascription():
    # Selecting position 0 is refused before anything descends, so neither the
    # wrapped name nor the type spelling is an expression point.
    program, table = program_of("(letrec (((ascribe# x @int) 1)) x)")

    for steps in [[1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 2]]:
        with pytest.raises(OccurrenceError):
            path_conditions(Path(steps), program, table)


def test_an_ascribed_binder_is_not_among_its_occurrences():
    # `occurrences` must not report the binding site, and every path it does
    # report must be one the walk accepts.
    program, table = program_of("(letrec (((ascribe# x @int) 1)) (if x 1 2))")

    found = occurrences(var("x", 0), program, table)

    assert Path([1, 0, 0]) not in found
    assert found == [Path([2, 1])]
    for path in found:
        path_conditions(path, program, table)


def test_a_define_binder_is_not_an_occurrence():
    # Promoting `define` to a core form created a new binder position, so it
    # needs the same refusal every other binding position has.
    program, table = program_of("(define x (if 1 2 3))")

    with pytest.raises(OccurrenceError):
        path_conditions(Path([1]), program, table)

    # The initializer beside it is an ordinary region.
    assert path_conditions(Path([2]), program, table) == []


def test_an_ascribed_define_binder_is_not_an_occurrence():
    program, table = program_of("(define (ascribe# x @int) (if 1 2 3))")

    for steps in [[1], [1, 1], [1, 2]]:
        with pytest.raises(OccurrenceError):
            path_conditions(Path(steps), program, table)


def test_a_define_binder_is_not_among_its_occurrences():
    program, table = program_of("(define x (if x 1 2))")

    found = occurrences(Symbol("x"), program, table)

    assert Path([1]) not in found
    assert found == [Path([2, 1])]
