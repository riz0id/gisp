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
from Syntax import Integer, Scope, SList, Symbol


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


# The two patterns a boolean `match` is written with. A clause is named by its
# pattern, so these are what `path_conditions` reports a boolean guard against
# — and each carries the empty product its constructor does, as the pattern for
# a constructor is written the way the construction is.
TRUE_PATTERN = SList([Symbol("#t"), SList([Symbol("prod#")])])
FALSE_PATTERN = SList([Symbol("#f"), SList([Symbol("prod#")])])


def consequent_of(test) -> list[Guard]:
    # The guards of the clause an `if`'s consequent became: the `(#t (prod#))`
    # pattern matched, which is the polarity the consequent's position carried.
    return [Guard(test=test, pattern=TRUE_PATTERN, consequent=True)]


def alternative_of(test) -> list[Guard]:
    # The guards of the clause an `if`'s alternative became. Two, because
    # reaching the second clause commits to both halves: `(#t (prod#))` was
    # tried and failed, and then `(#f (prod#))` matched. They read to the same
    # support, so the second is redundant — and stating it is what keeps the walk
    # faithful to a `match` of any width, where it is not.
    return [
        Guard(test=test, pattern=TRUE_PATTERN, consequent=False),
        Guard(test=test, pattern=FALSE_PATTERN, consequent=True),
    ]


# -- the two functions answer different questions ------------------------------


def test_sibling_ifs_give_one_list_but_four_distinct_occurrences():
    program, table = program_of(
        "(lambda (f x)"
        " (f (match (lt x 5) ((#t (prod#)) x) ((#f (prod#)) 0))"
        "    (match (lt 5 x) ((#t (prod#)) x) ((#f (prod#)) 0))))"
    )
    x = var("x", 0)

    below = lt(VarCExpr(x), IntegerCExpr(Integer(5)))
    above = lt(IntegerCExpr(Integer(5)), VarCExpr(x))

    # Keyed on the variable: one list, both tests, for the whole program.
    assert conditions(x, program, table) == [below, above]

    # Keyed on the occurrence: four references, four different answers. A
    # clause body sits at position 1 of its clause, so each use inside one is a
    # step deeper than the `if` it was written as.
    assert occurrences(x, program, table) == [
        Path([2, 1, 1, 1]),
        Path([2, 1, 2, 1]),
        Path([2, 2, 1, 2]),
        Path([2, 2, 2, 1]),
    ]
    assert path_conditions(Path([2, 1, 1, 1]), program, table) == []
    assert path_conditions(Path([2, 1, 2, 1]), program, table) == consequent_of(below)
    assert path_conditions(Path([2, 2, 1, 2]), program, table) == []
    assert path_conditions(Path([2, 2, 2, 1]), program, table) == consequent_of(above)


def test_conjoining_the_flat_list_is_unsatisfiable_where_no_occurrence_is():
    program, table = program_of(
        "(lambda (f x)"
        " (f (match (lt x 5) ((#t (prod#)) x) ((#f (prod#)) 0))"
        "    (match (lt 5 x) ((#t (prod#)) x) ((#f (prod#)) 0))))"
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
    program, table = program_of(
        "(lambda (x y) (match (lt y 5) ((#t (prod#)) x) ((#f (prod#)) 0)))"
    )
    x = var("x", 0)

    # The test never names `x`, so the harvest reports nothing for it.
    assert conditions(x, program, table) == []

    # The clause guards the use of `x` all the same, which is what the support
    # of `x` there depends on whenever `x` and `y` are dependent.
    assert occurrences(x, program, table) == [Path([2, 2, 1])]
    assert path_conditions(Path([2, 2, 1]), program, table) == consequent_of(
        lt(VarCExpr(var("y", 0)), IntegerCExpr(Integer(5)))
    )


def test_an_occurrence_in_the_alternative_carries_the_opposite_polarity():
    program, table = program_of(
        "(lambda (x) (match (lt x 5) ((#t (prod#)) 0) ((#f (prod#)) x)))"
    )
    x = var("x", 0)

    below = lt(VarCExpr(x), IntegerCExpr(Integer(5)))

    # The harvest hands back the scrutinee, which is the region the occurrence
    # is not in.
    assert conditions(x, program, table) == [below]

    assert path_conditions(Path([2, 3, 1]), program, table) == alternative_of(below)


def test_a_scrutinee_region_is_not_a_guard():
    program, table = program_of(
        "(lambda (x)"
        " (match (match (lt x 5) ((#t (prod#)) 1) ((#f (prod#)) 2))"
        "   ((#t (prod#)) 3)"
        "   ((#f (prod#)) x)))"
    )
    x = var("x", 0)

    harvested = conditions(x, program, table)

    # Both `match` forms mention `x`, so both scrutinees are harvested: the
    # outer one whole, and the inner one it contains.
    assert len(harvested) == 2
    assert harvested[1] == lt(VarCExpr(x), IntegerCExpr(Integer(5)))

    # Only the outer form guards the occurrence. The inner one sits in a
    # scrutinee region: every one of its clauses flows into the outer
    # scrutinee's value, so reaching `x` commits to none of them.
    assert path_conditions(Path([2, 3, 1]), program, table) == alternative_of(
        harvested[0]
    )


def test_nested_ifs_tighten_along_the_path():
    program, table = program_of(
        "(lambda (x)"
        " (match (lt x 10)"
        "   ((#t (prod#)) (match (lt 5 x) ((#t (prod#)) x) ((#f (prod#)) 0)))"
        "   ((#f (prod#)) 0)))"
    )
    x = var("x", 0)

    below = lt(VarCExpr(x), IntegerCExpr(Integer(10)))
    above = lt(IntegerCExpr(Integer(5)), VarCExpr(x))

    # Indistinguishable from the sibling case above: one flat list of two
    # tests, with nothing to say whether conjoining them is sound.
    assert conditions(x, program, table) == [below, above]

    # Here it is sound, and the walk says so by position.
    assert path_conditions(Path([2, 2, 1, 2, 1]), program, table) == consequent_of(
        below
    ) + consequent_of(above)

    # The use inside the inner scrutinee is past the outer guard but not the
    # inner.
    assert path_conditions(Path([2, 2, 1, 1, 2]), program, table) == consequent_of(
        below
    )


def test_a_letrec_initializer_is_walked_as_a_region():
    program, table = program_of(
        "(letrec ((y 1) (x (match (lt y 5) ((#t (prod#)) y) ((#f (prod#)) 0)))) 2)"
    )
    y = var("y", 0)

    # The binding position of `y` is not an occurrence; the two references in
    # the initializer are.
    assert occurrences(y, program, table) == [
        Path([1, 1, 1, 1, 1]),
        Path([1, 1, 1, 2, 1]),
    ]
    assert path_conditions(Path([1, 1, 1, 2, 1]), program, table) == consequent_of(
        lt(VarCExpr(y), IntegerCExpr(Integer(5)))
    )
    assert path_conditions(Path([1, 1, 1, 1, 1]), program, table) == []


def test_every_occurrence_has_a_path_the_walk_accepts():
    program, table = program_of(
        "(letrec ((f (lambda (x)"
        "              (match (lt x 5) ((#t (prod#)) x) ((#f (prod#)) (quote x))))))"
        " (match (lt 1 2) ((#t (prod#)) (f 3)) ((#f (prod#)) 4)))"
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
    program, table = program_of(
        "(lambda (x) (quote (match x ((#t (prod#)) 1) ((#f (prod#)) 2))))"
    )

    with pytest.raises(OccurrenceError):
        path_conditions(Path([2, 1]), program, table)


def test_a_path_off_the_tree_is_refused():
    program, table = program_of(
        "(lambda (x) (match (lt x 5) ((#t (prod#)) x) ((#f (prod#)) 0)))"
    )

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
    program, table = program_of(
        "(letrec (((ascribe# x @int) 1)) (match x ((#t (prod#)) 1) ((#f (prod#)) 2)))"
    )

    found = occurrences(var("x", 0), program, table)

    assert Path([1, 0, 0]) not in found
    assert found == [Path([2, 1])]
    for path in found:
        path_conditions(path, program, table)


def test_a_define_binder_is_not_an_occurrence():
    # Promoting `define` to a core form created a new binder position, so it
    # needs the same refusal every other binding position has.
    program, table = program_of(
        "(define x (match 1 ((#t (prod#)) 2) ((#f (prod#)) 3)))"
    )

    with pytest.raises(OccurrenceError):
        path_conditions(Path([1]), program, table)

    # The initializer beside it is an ordinary region.
    assert path_conditions(Path([2]), program, table) == []


def test_an_ascribed_define_binder_is_not_an_occurrence():
    program, table = program_of(
        "(define (ascribe# x @int) (match 1 ((#t (prod#)) 2) ((#f (prod#)) 3)))"
    )

    for steps in [[1], [1, 1], [1, 2]]:
        with pytest.raises(OccurrenceError):
            path_conditions(Path(steps), program, table)


def test_a_define_binder_is_not_among_its_occurrences():
    program, table = program_of(
        "(define x (match x ((#t (prod#)) 1) ((#f (prod#)) 2)))"
    )

    found = occurrences(Symbol("x"), program, table)

    assert Path([1]) not in found
    assert found == [Path([2, 1])]


# -- free variables --------------------------------------------------------------


def test_a_path_into_a_free_variable_is_refused():
    # Nothing within a `forall` is an expression: position 1 is a binder
    # position and a type, and the form is already the occurrence of what it
    # binds, so a path descending into one goes nowhere a guard applies to.
    program, table = program_of("(lt (forall (ascribe# n @int)) 5)")

    for path in [Path([1, 1]), Path([1, 1, 1]), Path([1, 0])]:
        with pytest.raises(OccurrenceError):
            path_conditions(path, program, table)


def test_the_path_to_a_free_variable_ends_at_the_form():
    # Which is why the occurrence is reported there and not inside the wrapper.
    program, table = program_of(
        "(lambda (x)"
        " (match (lt (forall (ascribe# n @int)) 5) ((#t (prod#)) x) ((#f (prod#)) 0)))"
    )

    assert occurrences(Symbol("n"), program, table) == [Path([2, 1, 1])]
    assert path_conditions(Path([2, 2, 1]), program, table) == consequent_of(
        lt(VarCExpr(Symbol("n")), IntegerCExpr(Integer(5)))
    )


def test_a_pattern_is_not_an_occurrence():
    # Position 0 of a clause is a binding position, so it takes the refusal
    # every other binding position takes — the one a letrec pair's name takes,
    # one level shallower.
    program, table = program_of("(match xs ((cons (prod# x r)) x))")

    with pytest.raises(OccurrenceError):
        path_conditions(Path([2, 0]), program, table)

    # The body beside it is an ordinary region, and the clause is reached by
    # the guard its pattern states.
    assert len(path_conditions(Path([2, 1]), program, table)) == 1


def test_a_pattern_binder_is_not_among_its_occurrences():
    program, table = program_of("(match xs ((cons (prod# x r)) x))")

    found = occurrences(var("x", 0), program, table)

    assert Path([2, 0, 1]) not in found
    assert found == [Path([2, 1])]

    for path in found:
        path_conditions(path, program, table)


def test_a_clause_guard_names_the_pattern_it_commits_to():
    # What `Guard.pattern` is for: a clause states which constructor the
    # scrutinee is, and a polarity alone cannot say that once there are more
    # than two clauses.
    program, table = program_of(
        "(match xs"
        " ((nil (prod#)) 0)"
        " ((cons (prod# x (nil (prod#)))) 1)"
        " ((cons (prod# x r)) 2))"
    )

    scrutinee = VarCExpr(Symbol("xs"))
    guards = path_conditions(Path([4, 1]), program, table)

    assert [guard.consequent for guard in guards] == [False, False, True]
    assert [guard.test for guard in guards] == [scrutinee] * 3
    assert [str(guard.pattern) for guard in guards] == [
        "(nil (prod#))",
        "(cons (prod# x (nil (prod#))))",
        "(cons (prod# x r))",
    ]
