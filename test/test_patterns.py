from collections.abc import Callable
from itertools import count

import pytest

from Analysis import refinements
from Analysis.Path import Path
from Analysis.Support import Support
from Evaluate import BUILTINS, Environment, EvalError, Value, evaluate
from Expander import expand, parse
from Reader import read
from Resolve import BindingTable, bindings
from Syntax import Expr, Integer, Scope, Symbol
from Syntax.Type import (
    BOOL,
    Context,
    FunType,
    IntType,
    ProdType,
    SumType,
    Type,
    constructor_arity,
    constructors,
    list_of,
)
from TypeCheck import TypeChecker, TypeCheckError
from TypeCheck.Patterns import operand_types

LOWER, UPPER = Integer.bounds()

# A constructor pattern is written exactly as the construction it inverts: one
# operand per field for a product, and the one payload for every other
# constructor. So `(#t (prod#))` and not `(#t)`, and `(cons (prod# x xs))` and
# not `(cons x xs)` — the abbreviated forms were a reading of the payload's
# fields where the constructor's own operands were what the pattern names.
#
# Counters from the 1500s, as `test_typecheck` takes the 1400s: no expectation
# here depends on a uid, but a collision with another file's hand-written
# scopes would be a confusing failure.


def counter(start: int) -> Callable[[], Scope]:
    uids = count(start)
    return lambda: Scope(next(uids))


def program_of(source: str, expand_at: int) -> tuple[Expr, BindingTable]:
    expanded = expand(read(source), fresh=counter(expand_at))
    program = parse(expanded, bindings(expanded))
    return program, bindings(program)


def listed_store() -> Context:
    # `xs` at a list of integers, which is what these patterns take apart.
    store: Context = Context.default()
    store.register(Symbol("xs"), list_of(IntType()))
    return store


def run(source: str) -> Value:
    # The unchecked path, where the evaluator answers for a pattern on its own.
    return evaluate(read(source), Environment(BUILTINS))


def narrowing(source: str, expand_at: int) -> Support:
    # What the analysis narrows the lambda's parameter to in the first clause's
    # body. The parameter is the one binder the lambda mints, so its scope is
    # the counter's first.
    program, table = program_of(source, expand_at)

    return refinements(
        Symbol("x", {Scope(expand_at)}),
        Path([2, 2]),
        program,
        table,
        Context.default(),
    )[-1]


# -- the abbreviated forms are refused -------------------------------------------


@pytest.mark.parametrize(
    ("source", "at"),
    [
        ("(match xs ((nil) 0) ((cons (prod# y ys)) 1))", 1500),
        ("(match xs ((nil (prod#)) 0) ((cons y ys) 1))", 1501),
    ],
)
def test_a_list_pattern_off_the_constructor_is_refused(source: str, at: int):
    # The two abbreviations: `(nil)` names no payload where the constructor
    # carries one, and `(cons y ys)` names two where it carries one.
    program, table = program_of(source, at)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table, store=listed_store()).check(program, IntType())


def test_a_list_pattern_written_in_full_checks():
    # The control for the two above, so neither refusal can be read as the
    # whole shape being refused.
    program, table = program_of(
        "(match xs ((nil (prod#)) 0) ((cons (prod# y ys)) 1))", 1502
    )

    TypeChecker(table=table, store=listed_store()).check(program, IntType())


def test_a_boolean_pattern_short_of_its_payload_is_refused():
    # `#t` carries the nullary product, which is exactly why the abbreviation
    # was invisible: reading the payload's fields gave zero operands, so a
    # boolean pattern looked like a bare constructor name.
    program, table = program_of("(match (le 1 2) ((#t) 1) ((#f) 2))", 1503)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table).check(program, IntType())

    program, table = program_of(
        "(match (le 1 2) ((#t (prod#)) 1) ((#f (prod#)) 2))", 1504
    )

    TypeChecker(table=table).check(program, IntType())


def test_a_product_pattern_is_written_with_its_fields():
    # The one constructor that is written as its fields, and so the one the
    # change leaves alone: `prod#` names two operands for a product of two,
    # because that is how the construction `(prod# 1 2)` is written too.
    store: Context = Context.default()
    store.register(Symbol("p"), ProdType([IntType(), IntType()]))

    program, table = program_of("(match p ((prod# a b) a))", 1505)
    TypeChecker(table=table, store=store).check(program, IntType())

    store = Context.default()
    store.register(Symbol("p"), ProdType([IntType(), IntType()]))

    program, table = program_of("(match p ((prod# q) 0))", 1506)
    with pytest.raises(TypeCheckError):
        TypeChecker(table=table, store=store).check(program, IntType())


# -- pattern and construction are one shape --------------------------------------


def test_a_pattern_takes_as_many_operands_as_the_construction_it_inverts():
    # The property the abbreviation violated, said over every sort that has
    # constructors at all rather than one test per sort. `constructor_arity` is
    # what `construction` checks an operand count against, and `operand_types`
    # is what `destructuring` checks one against — so this is the statement
    # that a constructor is written one way, whichever side it is written on.
    store: Context = Context.default()
    types: list[Type] = [
        BOOL,
        list_of(IntType()),
        ProdType(),
        ProdType([IntType(), IntType()]),
        ProdType({Symbol("x"): IntType(), Symbol("xs"): IntType()}),
        SumType({Symbol("zero"): ProdType(), Symbol("succ"): IntType()}),
    ]

    for ty in types:
        cons: dict[Symbol, Type] | None = constructors(ty, store)

        assert cons is not None

        for payload in cons.values():
            assert len(operand_types(ty, payload, store)) == constructor_arity(
                ty, store
            )


def test_a_sum_carrying_a_product_takes_one_operand_and_not_its_fields():
    # The arm the bug was in. A `cons` cell's payload is a product of two
    # fields, and reading those fields is what made `(cons x xs)` a pattern:
    # the type taken apart is a sum, so the count is one.
    store: Context = Context.default()
    cell: Type = SumType(
        {Symbol("pair"): ProdType([IntType(), IntType()]), Symbol("none"): ProdType()}
    )
    cons: dict[Symbol, Type] | None = constructors(cell, store)

    assert cons is not None
    assert operand_types(cell, cons[Symbol("pair")], store) == [
        ProdType([IntType(), IntType()])
    ]
    assert operand_types(cell, cons[Symbol("none")], store) == [ProdType()]


def test_a_list_type_is_read_through_its_unfolding():
    # `(@list t)` is a `@mu`, so the rule has to resolve it before it can tell
    # a sum from a product — and a `@mu` over a product has to resolve the
    # other way, or its pattern would be written as a payload it has none of.
    store: Context = Context.default()
    ty: Type = list_of(IntType())
    cons: dict[Symbol, Type] | None = constructors(ty, store)

    assert cons is not None
    assert [len(operand_types(ty, payload, store)) for payload in cons.values()] == [
        1,
        1,
    ]


# -- a payload binder -------------------------------------------------------------


def test_a_payload_binder_binds_the_product_the_constructor_carries():
    # One operand is what a sum constructor takes, so a bare binder there is a
    # well-formed pattern binding the whole payload — a shape no abbreviated
    # pattern could write. The inner `match` is what proves `c` is the cell's
    # product and not its first field.
    source = "(match xs ((nil (prod#)) 0) ((cons c) (match c ((prod# y ys) y))))"
    program, table = program_of(source, 1510)

    TypeChecker(table=table, store=listed_store()).check(program, IntType())


def test_a_payload_binder_covers_its_constructor():
    # And it covers the constructor whole, as any irrefutable pattern covers
    # the position it stands in: naming `cons` with a binder leaves no cell
    # unmatched, where naming it with `(prod# y (nil (prod#)))` leaves every
    # cell of two or more.
    program, table = program_of("(match xs ((nil (prod#)) 0) ((cons c) 1))", 1511)

    TypeChecker(table=table, store=listed_store()).check(program, IntType())

    program, table = program_of("(match xs ((cons c) 1))", 1512)

    with pytest.raises(TypeCheckError):
        TypeChecker(table=table, store=listed_store()).check(program, IntType())


# -- the evaluator ----------------------------------------------------------------


def test_the_evaluator_matches_the_payload_and_not_its_fields():
    assert run("(match '(1 2) ((nil (prod#)) 0) ((cons (prod# x xs)) x))") == Integer(1)
    assert run("(match '() ((nil (prod#)) 0) ((cons (prod# x xs)) x))") == Integer(0)

    # Two operands against the one payload a cell carries. The checker refuses
    # this first, and the evaluator refuses it too rather than matching the
    # fields of a payload the pattern did not name.
    with pytest.raises(EvalError):
        run("(match '(1 2) ((nil (prod#)) 0) ((cons x xs) x))")

    with pytest.raises(EvalError):
        run("(match (le 1 2) ((#t) 1) ((#f) 2))")


def test_the_evaluator_binds_a_payload_whole():
    source = "(match '(1 2) ((nil (prod#)) '()) ((cons c) (tail (cons c))))"

    assert run(source) == run("'(2)")


# -- the analysis -----------------------------------------------------------------


def test_a_boolean_guard_narrows_through_its_payload():
    # The reading that fails silently when it is missed: `Analysis.matching`
    # recognises a boolean clause by its pattern, so a payload operand it does
    # not expect makes every guard unrecognised and every support the whole of
    # the type — with no error raised anywhere.
    source = "(lambda (x) (match (lt x 5) ((#t (prod#)) x) ((#f (prod#)) 0)))"

    assert narrowing(source, 1520) == Support.of([(LOWER, 4)])


def test_a_payload_binder_guards_as_the_written_payload_does():
    # `(#t c)` is reached exactly when `(#t (prod#))` is, so it narrows the
    # same: the payload pattern is matched and passed over, which keeps the
    # reading from depending on how the empty product was written.
    source = "(lambda (x) (match (lt x 5) ((#t c) x) ((#f (prod#)) 0)))"

    assert narrowing(source, 1521) == Support.of([(LOWER, 4)])


# -- the libraries ----------------------------------------------------------------


def test_a_lambda_over_a_list_checks_with_the_written_patterns():
    # The shape both libraries are written in, checked here so a regression in
    # the rule is reported by this file and not only by the driver.
    source = (
        "(lambda ((ascribe# xs (@list @int)))"
        " (match xs"
        "   ((nil (prod#)) 0)"
        "   ((cons (prod# x (nil (prod#)))) x)"
        "   ((cons (prod# x (cons (prod# y ys)))) (+ x y))))"
    )
    program, table = program_of(source, 1530)
    ty: FunType = FunType(
        return_type=IntType(),
        argument_type=ProdType({Symbol("xs"): list_of(IntType())}),
    )

    TypeChecker(table=table).check(program, ty)
