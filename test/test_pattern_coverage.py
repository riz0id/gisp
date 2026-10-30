from collections.abc import Callable
from itertools import count

import pytest

from Analysis import coverages, matches, refinements, unreachable
from Analysis.Path import Path
from Analysis.PatternCoverage import (
    CoverageError,
    clause_pattern,
    clauses,
    coverage,
)
from Expander import expand, parse
from Reader import read
from Resolve import BindingTable, bindings
from Syntax import Expr, Scope, Symbol
from Syntax.Type import BOOL, Context, IntType, ProdType, Type, list_of
from TypeCheck import TypeChecker, TypeCheckError
from TypeCheck.Patterns import exhaustive, missing, useful

# Coverage is one relation read two ways. A row of patterns is *useful* against
# a matrix when it matches a value the matrix does not: a `match` is complete
# when a row of wildcards is useful against none of its clauses, and a clause is
# redundant when it is not useful against the clauses before it. `missing` is
# the constructive half — it answers the row, which is a pattern, so what a
# refusal prints is the clause that repairs it.
#
# Counters from the 1900s, as `test_toplevel` takes the 1800s.

LIST: Type = list_of(IntType())


def counter(start: int) -> Callable[[], Scope]:
    uids = count(start)
    return lambda: Scope(next(uids))


def program_of(source: str, expand_at: int) -> tuple[Expr, BindingTable]:
    expanded = expand(read(source), fresh=counter(expand_at))
    program = parse(expanded, bindings(expanded))
    return program, bindings(program)


def listed_store() -> Context:
    # The scrutinees these matches are written over, ascribed by hand: these
    # programs are given to the analysis directly and never to `check`, so
    # nothing has put a binder's type in the store for them.
    store: Context = Context.default()
    store.register(Symbol("xs"), LIST)
    store.register(Symbol("ys"), LIST)
    store.register(Symbol("b"), BOOL)
    store.register(Symbol("n"), IntType())
    store.register(Symbol("p"), ProdType([IntType(), IntType()]))
    return store


def matrix_of(source: str, expand_at: int) -> tuple[list[list[Expr]], BindingTable]:
    # A `match`'s clause patterns as the one-column matrix `coverage` builds,
    # with the table they resolve against. Read off a parsed program rather
    # than written by hand, because a constructor is named by resolution and
    # not by its identifier.
    program, table = program_of(source, expand_at)
    read_clauses: tuple[Expr, list[Expr]] | None = clauses(program)

    assert read_clauses is not None

    return [[clause_pattern(clause)] for clause in read_clauses[1]], table


def witness_of(source: str, expand_at: int, ty: Type) -> Expr:
    rows, table = matrix_of(source, expand_at)
    found: list[Expr] | None = missing(rows, [ty], listed_store(), table)

    assert found is not None

    return found[0]


# -- missing is exhaustive made constructive -------------------------------------


COLUMNS: list[tuple[str, int, Type, bool]] = [
    ("(match xs ((nil (prod#)) 0) ((cons (prod# y ys)) 1))", 1900, LIST, True),
    ("(match xs ((cons (prod# y ys)) 1))", 1902, LIST, False),
    ("(match xs ((nil (prod#)) 0))", 1904, LIST, False),
    (
        "(match xs ((nil (prod#)) 0) ((cons (prod# y (nil (prod#)))) 1))",
        1906,
        LIST,
        False,
    ),
    ("(match xs ((nil (prod#)) 0) ((cons c) 1))", 1908, LIST, True),
    ("(match b ((#t (prod#)) 0) ((#f (prod#)) 1))", 1910, BOOL, True),
    ("(match b ((#t (prod#)) 0))", 1912, BOOL, False),
    ("(match b (c 0))", 1914, BOOL, True),
    ("(match n (1 0) (m 2))", 1916, IntType(), True),
    ("(match n (1 0))", 1918, IntType(), False),
    ("(match p ((prod# a c) 0))", 1920, ProdType([IntType(), IntType()]), True),
]


@pytest.mark.parametrize(("source", "at", "ty", "complete"), COLUMNS)
def test_missing_answers_what_exhaustive_decides(
    source: str, at: int, ty: Type, complete: bool
):
    # The one walk, asked the two ways. This is what fails if `exhaustive` is
    # ever given a reading of its own again.
    rows, table = matrix_of(source, at)
    store: Context = listed_store()

    assert exhaustive(rows, [ty], store, table) == complete
    assert (missing(rows, [ty], store, table) is None) == complete


# -- the witness ------------------------------------------------------------------


def test_the_witness_names_the_constructor_no_clause_names():
    assert str(witness_of("(match b ((#t (prod#)) 0))", 1922, BOOL)) == "(#f _)"
    assert str(witness_of("(match xs ((cons (prod# y ys)) 1))", 1923, LIST)) == (
        "(nil _)"
    )


def test_the_witness_reaches_the_depth_the_clauses_wrote():
    # Both constructors of a list are named and every list of two or more is
    # still unmatched, so the witness is a cell whose rest is a cell — which is
    # the recursion `missing` makes and a set of constructor names could not.
    source = "(match xs ((nil (prod#)) 0) ((cons (prod# y (nil (prod#)))) 1))"

    assert str(witness_of(source, 1924, LIST)) == "(cons (prod# _ (cons _)))"


def test_the_witness_for_a_ground_column_is_a_bare_wildcard():
    # `@int` has a constructor per value, so what a run of literals leaves
    # unmatched is "some other integer" — and `_` is how that is written.
    assert str(witness_of("(match n (1 0))", 1925, IntType())) == "_"


@pytest.mark.parametrize(
    ("source", "at", "ty"),
    [
        ("(match b ((#t (prod#)) 0))", 1926, BOOL),
        ("(match xs ((cons (prod# y ys)) 1))", 1927, LIST),
        ("(match xs ((nil (prod#)) 0) ((cons (prod# y (nil (prod#)))) 1))", 1928, LIST),
        ("(match n (1 0))", 1929, IntType()),
    ],
)
def test_a_witness_is_the_clause_that_repairs_the_match(source: str, at: int, ty: Type):
    # A witness is a pattern and not a value, which is the whole of why it is
    # worth printing: adding it as a clause is what completes the `match` it
    # was reported about. `_` is a binder, so a wildcard in it matches
    # everything the position leaves open.
    rows, table = matrix_of(source, at)
    store: Context = listed_store()
    found: list[Expr] | None = missing(rows, [ty], store, table)

    assert found is not None
    assert exhaustive([*rows, found], [ty], store, table)


# -- usefulness, and the clause that is worth writing ----------------------------


def test_a_repeated_clause_is_useful_against_nothing():
    source = (
        "(match xs ((nil (prod#)) 0) ((cons (prod# y ys)) 1) ((cons (prod# z zs)) 2))"
    )
    rows, table = matrix_of(source, 1930)
    store: Context = listed_store()

    assert useful(rows[:1], rows[1], [LIST], store, table)
    assert not useful(rows[:2], rows[2], [LIST], store, table)


def test_a_wider_clause_is_useful_after_a_narrower_one():
    # The payload binder covers every cell, and the clause before it covers
    # only the one-element ones — so there is a value left for it to match.
    source = (
        "(match xs ((nil (prod#)) 0) ((cons (prod# y (nil (prod#)))) 1) ((cons c) 2))"
    )
    rows, table = matrix_of(source, 1932)

    assert useful(rows[:2], rows[2], [LIST], listed_store(), table)


def test_a_narrower_clause_is_useless_after_a_wider_one():
    source = (
        "(match xs ((nil (prod#)) 0) ((cons c) 1) ((cons (prod# y (nil (prod#)))) 2))"
    )
    rows, table = matrix_of(source, 1934)

    assert not useful(rows[:2], rows[2], [LIST], listed_store(), table)


def test_a_repeated_literal_is_useful_against_nothing():
    # The arm coverage has no use for: a literal never completes a signature,
    # so `missing` drops it, and this is the one reading that compares a run of
    # literals against itself.
    rows, table = matrix_of("(match n (1 0) (1 2) (m 3))", 1936)
    store: Context = listed_store()

    assert not useful(rows[:1], rows[1], [IntType()], store, table)
    assert useful(rows[:2], rows[2], [IntType()], store, table)


def test_a_distinct_literal_is_useful():
    rows, table = matrix_of("(match n (1 0) (2 3) (m 4))", 1938)

    assert useful(rows[:1], rows[1], [IntType()], listed_store(), table)


# -- coverage of one match -------------------------------------------------------


def test_coverage_reads_a_complete_match():
    source = "(match xs ((nil (prod#)) 0) ((cons (prod# y ys)) 1))"
    program, table = program_of(source, 1940)
    verdict = coverage(program, table, listed_store())

    assert verdict is not None
    assert verdict.complete()
    assert verdict.witness is None
    assert verdict.redundant == ()
    assert verdict.scrutinee == LIST


def test_coverage_names_the_witness_of_an_incomplete_match():
    program, table = program_of("(match xs ((cons (prod# y ys)) 1))", 1942)
    verdict = coverage(program, table, listed_store())

    assert verdict is not None
    assert not verdict.complete()
    assert str(verdict.witness) == "(nil _)"


def test_coverage_names_the_clauses_that_are_redundant():
    source = (
        "(match xs"
        " ((nil (prod#)) 0)"
        " ((cons (prod# y ys)) 1)"
        " ((cons (prod# z (nil (prod#)))) 2))"
    )
    program, table = program_of(source, 1944)
    verdict = coverage(program, table, listed_store())

    assert verdict is not None
    assert verdict.complete()
    assert verdict.redundant == (2,)


def test_coverage_declines_where_the_scrutinees_type_cannot_be_read():
    # Coverage is a question about a type — which constructors there are to
    # name — so a `match` with no type to read has nothing to be covered.
    for source, at in [
        ("(match (lambda (y) 1) ((#t (prod#)) 0) ((#f (prod#)) 1))", 1946),
        ("(match zs ((nil (prod#)) 0) ((cons (prod# y ys)) 1))", 1948),
    ]:
        program, table = program_of(source, at)

        assert coverage(program, table, listed_store()) is None


def test_coverage_declines_where_a_clause_is_not_a_pattern_for_the_type():
    # A `#t` pattern on an `@int` scrutinee is a type error. Reporting it as a
    # coverage gap would have the analysis refuse the program on the checker's
    # behalf, in a phase that runs after it and in words about the wrong thing.
    program, table = program_of("(match n ((#t (prod#)) 2) ((#f (prod#)) 3))", 1950)

    assert coverage(program, table, listed_store()) is None


def test_coverage_declines_where_a_pattern_takes_the_wrong_operands():
    # The same refusal, one level down, and the one that makes the rows safe to
    # walk at all: `missing` reads a constructor pattern's operands as the
    # constructor's own, and a pattern written with the wrong count of them
    # would leave a row shorter than its columns.
    program, table = program_of("(match xs ((nil (prod#)) 0) ((cons y ys) 1))", 1952)

    assert coverage(program, table, listed_store()) is None


def test_reading_a_type_does_not_write_the_store():
    # What `probing` is for. `type_of` and `pattern` register the binders they
    # read through, and the analysis is asking a question rather than checking
    # a program.
    source = "(match xs ((nil (prod#)) 0) ((cons (prod# y ys)) y))"
    program, table = program_of(source, 1954)
    store: Context = listed_store()
    size: int = len(store)

    assert coverage(program, table, store) is not None
    assert len(store) == size


# -- every match in a program ----------------------------------------------------


def test_matches_finds_a_nested_match_as_well_as_its_own():
    source = (
        "(match xs"
        " ((nil (prod#)) 0)"
        " ((cons (prod# y ys))"
        "  (match ys ((nil (prod#)) 1) ((cons (prod# z zs)) 2))))"
    )
    program, _ = program_of(source, 1956)

    assert matches(program) == [Path([]), Path([3, 1])]


def test_matches_leaves_a_match_inside_quoted_data_alone():
    # Quoted syntax is data and not a form, which is the reading `arms` already
    # makes and this one inherits.
    source = (
        "(match xs ((nil (prod#)) (quote (match n (1 2)))) ((cons (prod# y ys)) 1))"
    )
    program, _ = program_of(source, 1958)

    assert matches(program) == [Path([])]


def test_coverages_answers_for_each_match_in_order():
    source = (
        "(match xs"
        " ((nil (prod#)) 0)"
        " ((cons (prod# y r))"
        # On `xs` again, and not on the rest the pattern bound: a pattern's
        # binder is ascribed by whoever checked the program, and these are
        # given to the analysis directly — so the inner `match` would have no
        # type to read and no verdict to give.
        "  (match xs ((cons (prod# z zs)) 2))))"
    )
    program, table = program_of(source, 1960)
    found = coverages(program, table, listed_store())

    assert [path for path, _ in found] == [Path([]), Path([3, 1])]
    assert found[0][1] is not None and found[0][1].complete()
    assert found[1][1] is not None and not found[1][1].complete()


# -- the precondition the analysis is read under ---------------------------------


def test_refinements_refuses_a_match_that_leaves_a_value_unmatched():
    # A guard says that reaching a clause means the clauses before it failed
    # and this one matched, which describes the program only where some clause
    # always matches.
    program, table = program_of("(match xs ((cons (prod# y ys)) y))", 1962)

    with pytest.raises(CoverageError):
        refinements(Symbol("n"), Path([2]), program, table, listed_store())


def test_unreachable_refuses_a_match_that_leaves_a_value_unmatched():
    program, table = program_of("(match xs ((cons (prod# y ys)) y))", 1964)

    with pytest.raises(CoverageError):
        unreachable(Path([2]), program, table, listed_store())


def test_the_analysis_is_not_refused_where_coverage_declines():
    # The guard refuses an incomplete `match` and not an unanalysable one.
    # Declining to answer is not an answer of "incomplete", and a program the
    # checker would refuse is still one the walk reads.
    for source, at in [
        ("(match n ((#t (prod#)) 2) ((#f (prod#)) 3))", 1966),
        ("(match zs ((cons (prod# y ys)) 1))", 1968),
    ]:
        program, table = program_of(source, at)

        assert not unreachable(Path([2]), program, table, listed_store())


def test_the_analysis_reads_a_complete_match():
    program, table = program_of("(match b ((#t (prod#)) 1) ((#f (prod#)) 0))", 1970)
    store: Context = listed_store()

    assert not unreachable(Path([2]), program, table, store)
    assert not unreachable(Path([3]), program, table, store)


# -- a redundant clause is unreachable -------------------------------------------


REDUNDANT: str = (
    "(match xs"
    " ((nil (prod#)) 0)"
    " ((cons (prod# y ys)) 1)"
    " ((cons (prod# z (nil (prod#)))) 2))"
)


def test_a_redundant_clause_is_unreachable():
    # The structural reason beside the contradictory one: no value reaches the
    # clause, whatever any variable may be. A guard could not say it — the
    # scrutinee is a list, and a `Support` is a set of integers.
    program, table = program_of(REDUNDANT, 1972)
    store: Context = listed_store()

    assert unreachable(Path([4]), program, table, store)
    assert unreachable(Path([4, 1]), program, table, store)


def test_the_clauses_beside_a_redundant_one_are_reached():
    program, table = program_of(REDUNDANT, 1974)
    store: Context = listed_store()

    assert not unreachable(Path([2]), program, table, store)
    assert not unreachable(Path([3]), program, table, store)
    assert not unreachable(Path([3, 1]), program, table, store)


def test_a_redundant_clause_refines_as_it_did():
    # Coverage says which clause is reached and changes nothing about what a
    # variable is narrowed to: the series for a list clause is the ground range
    # throughout, as it was before coverage was read at all.
    program, table = program_of(REDUNDANT, 1976)
    series = refinements(Symbol("n"), Path([3, 1]), program, table, listed_store())

    assert series == [series[0]] * len(series)


# -- the refusal names what is missing -------------------------------------------


def test_the_type_error_names_the_pattern_that_is_missing():
    program, table = program_of("(match xs ((cons (prod# y ys)) y))", 1978)
    checker = TypeChecker(table=table, store=listed_store())

    with pytest.raises(TypeCheckError, match=r"leaves \(nil _\) unmatched"):
        checker.check(program, IntType())
