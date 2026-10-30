from collections.abc import Callable
from itertools import count

import pytest

from Analysis.Measure import Constraints, Edge, box, extensions, probability, total
from Analysis.PatternCoverage import CoverageError
from Analysis.Shape import (
    CallShape,
    ConShape,
    GroundShape,
    OpaqueShape,
    ProductShape,
    Shape,
    Verdict,
    admits,
    naming,
    opened,
    topped,
)
from Analysis.Support import Support
from Analysis.Walk import (
    Branch,
    Call,
    Summary,
    applied,
    elements,
    procedures,
    summaries,
    weight,
)
from Expander import expand, parse
from Reader import read, read_all
from Resolve import BindingTable, bindings
from Syntax import Expr, Scope, Symbol
from Syntax.Type import BOOL, Context, IntType, ProdType, Type, list_of
from TypeCheck import TypeChecker, check_toplevel
from TypeCheck.Patterns import WILDCARD

# A procedure's walks are computed once per *class* of its argument — once per
# way that argument can be written — and a call is answered by reference rather
# than followed. So the summary of a program is finite however deep its
# recursion goes, and there is no bound to choose and nothing to truncate.
#
# Within one class, a branch's weight is a count of orderings of that class's
# ground binders: `pass`'s third class binds two elements, so its comparison
# splits the two orderings one each. A branch decided by what a *call* returns
# has no weight at all, and says so rather than guessing one.
#
# Counters from the 2000s, as `test_pattern_coverage` takes the 1900s.

LIST: Type = list_of(IntType())

PASS: str = """
(define (ascribe# pass (@fun (@prod (@list @int)) (@list @int)))
  (lambda ((ascribe# xs (@list @int)))
    (match xs
      ((nil (prod#)) xs)
      ((cons (prod# x (nil (prod#)))) xs)
      ((cons (prod# x (cons (prod# y ys))))
       (match (le x y)
         ((#t (prod#)) (cons (prod# x (pass (cons (prod# y ys))))))
         ((#f (prod#)) (cons (prod# y (pass (cons (prod# x ys)))))))))))
"""

SORT: str = """
(define (ascribe# sort (@fun (@prod (@list @int)) (@list @int)))
  (lambda ((ascribe# xs (@list @int)))
    (letrec (((ascribe# ys (@list @int)) (pass xs)))
      (match (eq xs ys)
        ((#t (prod#)) xs)
        ((#f (prod#)) (sort ys))))))
"""


def counter(start: int) -> Callable[[], Scope]:
    uids = count(start)
    return lambda: Scope(next(uids))


def analysed(source: str, at: int) -> tuple[list[tuple[Expr, BindingTable]], Context]:
    # The whole source through the phases that precede evaluation, each form's
    # table kept beside it. The driver's `analysed`, written here so the tests
    # do not reach into the executable for it.
    store: Context = Context.default()
    forms: list[tuple[Expr, BindingTable]] = []

    for form in read_all(source):
        expanded: Expr = expand(form, fresh=counter(at))
        program: Expr = parse(expanded, bindings(expanded))
        table: BindingTable = bindings(program)

        check_toplevel(program, TypeChecker(table=table, store=store))
        forms.append((program, table))
        at += 100

    return forms, store


def summarised(source: str, at: int) -> dict[Symbol, Summary | None]:
    forms, store = analysed(source, at)

    return summaries(forms, store)


def at(summary: Summary, written: str) -> tuple[int, ...]:
    # The key of the class written as `written`.
    #
    # By how a class is written and never by its position: the order follows
    # the type's constructor order, which is not a fact worth resting a test
    # on.
    for key in summary.branches:
        if " ".join(
            str(summary.classes[position][index])
            for position, index in enumerate(key)
        ) == written:
            return key

    raise AssertionError(f"no class written {written}: {summary.classes}")


def shares(branches: tuple[Branch, ...]) -> list[float | None]:
    over = elements(branches)

    return [probability(weight(branch, over), over) for branch in branches]


def program_of(source: str, at: int) -> tuple[Expr, BindingTable]:
    expanded = expand(read(source), fresh=counter(at))
    program = parse(expanded, bindings(expanded))
    return program, bindings(program)


def carried(shape: Shape) -> list[Shape]:
    # Every shape within one, so a test can ask what a result holds.
    match shape:
        case ConShape(payload=payload):
            return [shape, *carried(payload)]
        case ProductShape(fields=fields):
            return [shape, *(held for field in fields for held in carried(field))]
        case _:
            return [shape]


# -- the measure ---------------------------------------------------------------


def test_an_unconstrained_ordering_admits_every_permutation():
    over = (Symbol("a"), Symbol("b"), Symbol("c"))

    assert extensions(Constraints(), over) == 6 == total(over)


def test_a_chain_admits_one_ordering():
    a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
    chained = Constraints(
        order=(Edge(a, b, True), Edge(b, c, True)),
    )

    assert extensions(chained, (a, b, c)) == 1


def test_one_comparison_halves_the_orderings():
    a, b, c = Symbol("a"), Symbol("b"), Symbol("c")

    assert extensions(Constraints(order=(Edge(a, b, True),)), (a, b, c)) == 3


def test_the_two_ways_of_one_comparison_sum_to_every_ordering():
    a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
    over = (a, b, c)
    taken = Constraints(order=(Edge(a, b, False),))
    failed = Constraints(order=(Edge(b, a, True),))

    assert extensions(taken, over) + extensions(failed, over) == total(over)


def test_a_cycle_admits_no_ordering():
    # The relational counterpart of an empty support, and the reason an
    # infeasible walk weighs zero without being special-cased.
    a, b = Symbol("a"), Symbol("b")
    cyclic = Constraints(order=(Edge(a, b, True), Edge(b, a, True)))

    assert extensions(cyclic, (a, b)) == 0
    assert cyclic.infeasible()


def test_equality_between_two_distinct_elements_admits_no_ordering():
    # What `eq` commits to when it is taken: each below the other. No ordering
    # of distinct keys satisfies it, so the branch weighs zero.
    a, b = Symbol("a"), Symbol("b")
    equal = Constraints(order=(Edge(a, b, False), Edge(b, a, False)))

    assert extensions(equal, (a, b)) == 0


def test_an_edge_outside_the_elements_is_not_measured():
    # The count would no longer be a count of *input* orderings, so the shares
    # would stop summing to one.
    a, b = Symbol("a"), Symbol("b")
    outside = Constraints(order=(Edge(a, b, True),))

    assert extensions(outside, (a,)) is None


def test_a_narrowed_element_is_not_measured():
    # An ordering says which element is larger. It cannot say that one of them
    # is three.
    a = Symbol("a")
    narrowed = Constraints(supports={a: Support.point(3)})

    assert extensions(narrowed, (a,)) is None


def test_the_unary_reading_is_a_product_of_cardinalities():
    a, b = Symbol("a"), Symbol("b")
    bounded = Constraints(supports={a: Support.of([(0, 9)]), b: Support.of([(0, 4)])})

    assert box(bounded, (a, b)) == 50


def test_a_probability_is_a_share_of_the_orderings():
    a, b = Symbol("a"), Symbol("b")

    assert probability(1, (a, b)) == 0.5
    assert probability(None, (a, b)) is None


# -- the fan-out is the type's summands ----------------------------------------


def test_opening_a_list_at_cons_gives_its_one_payload():
    # The algebraic step: ⊤ at a type opened into one constructor of it, with
    # one unknown per column `operand_types` names.
    #
    # A sum constructor contributes exactly one column — the payload it carries
    # — however that payload is written. So a cons cell opens to a single ⊤ at
    # the product of an element and a rest, and is *not* spread here: the
    # `(prod# x xs)` pattern written for the payload is what spreads it, which
    # is why a list is unfolded two levels per cell rather than one.
    found = opened(Symbol("cons"), OpaqueShape(LIST), Context.default(), naming())

    assert found is not None

    whole, certain = found

    # Two constructors, so committing to one of them is not certain.
    assert certain is False

    match whole:
        case ConShape(constructor=named, payload=OpaqueShape(ty=payload)):
            assert named == Symbol("cons")
            assert isinstance(payload, ProdType)
            assert list(payload.types()) == [IntType(), LIST]
        case _:
            raise AssertionError(f"not a cons cell: {whole}")


def test_opening_a_list_at_nil_gives_the_empty_product():
    found = opened(Symbol("nil"), OpaqueShape(LIST), Context.default(), naming())

    assert found is not None
    assert found[0] == ConShape(Symbol("nil"), OpaqueShape(ProdType()), LIST)


def test_the_product_pattern_is_what_spreads_a_cells_payload():
    # The second of the two levels. `(cons (prod# x xs))` binds an element and
    # a rest, and they come from the payload being opened by the product
    # pattern — not from the cell having been opened into two columns.
    store = Context.default()
    pattern, table = program_of("(cons (prod# x xs))", 2040)
    found = admits(pattern, OpaqueShape(LIST), table, store, naming())

    assert found.verdict is Verdict.MAYBE

    bound = {binder.identifier: shape for binder, shape in found.bound.items()}

    assert isinstance(bound["x"], GroundShape)
    assert bound["xs"] == OpaqueShape(LIST)


def test_opening_a_boolean_names_one_of_two_constructors():
    store = Context.default()

    assert opened(Symbol("#t"), OpaqueShape(BOOL), store, naming()) is not None
    assert opened(Symbol("#f"), OpaqueShape(BOOL), store, naming()) is not None


def test_a_constructor_the_type_does_not_have_opens_nothing():
    found = opened(Symbol("nil"), OpaqueShape(BOOL), Context.default(), naming())

    assert found is None


def test_committing_to_a_products_one_constructor_is_certain():
    # A product has one constructor, so `(prod# x xs)` refutes nothing and
    # always matches; a sum of two is a coin a clause names one side of.
    pair = ProdType([IntType(), IntType()])
    found = opened(Symbol("prod#"), OpaqueShape(pair), Context.default(), naming())

    assert found is not None
    assert found[1] is True


def test_every_clause_of_a_complete_match_admits_an_unknown_list():
    # The three clauses `pass` writes, read against ⊤ at the list type. None is
    # refuted and none is certain, so each is a branch — which is what makes a
    # walk set rather than a single walk.
    store = Context.default()
    fresh = naming()
    shape: Shape = OpaqueShape(LIST)

    for source in [
        "(nil (prod#))",
        "(cons (prod# x (nil (prod#))))",
        "(cons (prod# x (cons (prod# y ys))))",
    ]:
        pattern, table = program_of(source, 2000)

        assert admits(pattern, shape, table, store, fresh).verdict is Verdict.MAYBE


def test_a_clause_naming_another_constructor_is_refuted():
    store = Context.default()
    fresh = naming()
    found = opened(Symbol("nil"), OpaqueShape(LIST), store, fresh)

    assert found is not None

    empty, _ = found
    pattern, table = program_of("(cons (prod# x xs))", 2010)

    assert admits(pattern, empty, table, store, fresh).verdict is Verdict.NO


# -- reading a quoted argument -------------------------------------------------


def test_a_quoted_list_is_read_one_level():
    # One level and no deeper: the argument is a `cons`, and what its tail is
    # does not enter into it. Reading the spine is what would make an answer
    # depend on a length.
    datum, table = program_of("(quote (3 1 2))", 2020)
    shape = topped(datum.items[1], LIST, Context.default(), naming())

    # `(cons _)` and not `(cons (prod# ...))`: a sum constructor carries one
    # payload, so one level is that payload left unknown.
    assert str(shape.to_pattern()) == "(cons _)"


def test_two_quoted_lists_of_different_lengths_read_alike():
    store = Context.default()
    one, _ = program_of("(quote (1))", 2030)
    many, _ = program_of("(quote (3 1 4 1 5 9 2 6))", 2040)

    assert str(
        topped(one.items[1], LIST, store, naming()).to_pattern()
    ) == str(topped(many.items[1], LIST, store, naming()).to_pattern())


def test_the_empty_list_reads_as_nil():
    datum, table = program_of("(quote ())", 2050)
    shape = topped(datum.items[1], LIST, Context.default(), naming())

    assert str(shape.to_pattern()) == "(nil _)"


# -- the summary of a procedure ------------------------------------------------


def test_a_procedure_is_summarised_once_per_class():
    # `pass` writes three clauses on its argument, so three classes; `sort`
    # never matches its argument, so two. One entry each, and that is the whole
    # summary — there is no continuation and nothing deferred.
    found = summarised(PASS + SORT, 2100)
    passed = found[Symbol("pass")]
    sorted_ = found[Symbol("sort")]

    assert passed is not None and sorted_ is not None
    assert len(passed.branches) == 3
    assert len(sorted_.branches) == 2
    assert {
        " ".join(str(written) for written in group) for group in passed.classes
    } == {"(cons (prod# _ (cons _))) (cons (prod# _ (nil _))) (nil _)"}


def test_a_class_that_decides_its_clause_has_one_branch():
    # The empty list and the one-element list each reach one clause and make no
    # comparison, so each is a single branch taken by everything in its class.
    found = summarised(PASS, 2110)
    passed = found[Symbol("pass")]

    assert passed is not None

    for written in ["(nil _)", "(cons (prod# _ (nil _)))"]:
        branches = passed.branches[at(passed, written)]

        assert len(branches) == 1
        assert branches[0].calls == ()
        assert shares(branches) == [1.0]


def test_the_comparison_within_a_class_splits_it_evenly():
    # `pass`'s third class binds two elements and compares them, so the two
    # orderings of those elements go one to each branch. Exact, and it needs
    # nothing of the call the branch goes on to make.
    found = summarised(PASS, 2120)
    passed = found[Symbol("pass")]

    assert passed is not None

    branches = passed.branches[at(passed, "(cons (prod# _ (cons _)))")]

    assert len(branches) == 2
    assert shares(branches) == [0.5, 0.5]
    assert sum(shares(branches)) == 1.0


def test_a_recursive_call_is_a_reference_and_not_a_traversal():
    # The whole claim. `pass`'s third class calls `pass`, and what the branch
    # answers holds a reference to `pass` at the classes its argument can be —
    # a cons whose tail is unknown is a one-element list or a longer one, so
    # classes 1 and 2. Nothing was traversed to find that out.
    found = summarised(PASS, 2130)
    passed = found[Symbol("pass")]

    assert passed is not None

    passing = at(passed, "(cons (prod# _ (cons _)))")
    deeper = at(passed, "(cons (prod# _ (nil _)))")

    for branch in passed.branches[passing]:
        # A cons whose tail is unknown is a one-element list or a longer one,
        # so the call falls in exactly those two classes — and neither was
        # traversed to find that out.
        [call] = branch.calls

        assert call.procedure == Symbol("pass")
        assert set(call.classes[0]) == {passing[0], deeper[0]}
        assert any(
            isinstance(held, CallShape) and held.procedure == Symbol("pass")
            for held in carried(branch.result)
        )


def test_the_call_graph_has_the_programs_cycles_in_it():
    # `pass` calls itself and `sort` calls both, so the graph has a self-loop
    # at each — which is what recursion comes out as when it is not followed.
    found = summarised(PASS + SORT, 2140)
    edges = {
        (binary.procedure, call.procedure)
        for binary in found.values()
        if binary is not None
        for branches in binary.branches.values()
        for branch in branches
        for call in branch.calls
    }

    assert (Symbol("pass"), Symbol("pass")) in edges
    assert (Symbol("sort"), Symbol("pass")) in edges
    assert (Symbol("sort"), Symbol("sort")) in edges


def test_a_branch_decided_by_a_calls_result_has_no_weight():
    # `sort` branches on `(eq xs ys)` where `ys` is what `pass` returned, and
    # this reads a call by reference — so it knows nothing of that result. The
    # branch is reported with no probability rather than with a wrong one.
    found = summarised(PASS + SORT, 2150)
    sorted_ = found[Symbol("sort")]

    assert sorted_ is not None

    branches = sorted_.branches[at(sorted_, "(cons _)")]

    assert len(branches) == 2
    assert all(not branch.measured for branch in branches)
    assert shares(branches) == [None, None]


# -- nothing is bounded -------------------------------------------------------


def test_the_analysis_has_no_budget_to_set():
    # The claim as a fact about the module: there is nothing to tune, because
    # there is no unrolling to bound.
    import Analysis.Walk as walk

    for gone in ["Budget", "BUDGET", "Tally", "walks_of", "entered", "capped"]:
        assert not hasattr(walk, gone), gone


def test_no_branch_is_ever_truncated():
    found = summarised(PASS + SORT, 2160)

    for binary in found.values():
        assert binary is not None

        for branches in binary.branches.values():
            assert branches
            for branch in branches:
                assert not hasattr(branch, "truncated")


# -- a form that applies a procedure ------------------------------------------


def test_a_form_is_answered_as_the_class_its_argument_falls_in():
    forms, store = analysed(PASS + SORT + "(sort (quote (3 1 4 1 5 9 2 6)))", 2200)
    form, table = forms[-1]
    found = applied(form, table, store, procedures(forms, store))

    assert found is not None

    procedure, indices = found
    sorted_ = summarised(PASS + SORT + "(sort (quote (3 1 4 1 5 9 2 6)))", 2201)[
        Symbol("sort")
    ]

    assert sorted_ is not None
    assert procedure == Symbol("sort")
    assert indices == (at(sorted_, "(cons _)"),)


def test_the_literals_length_does_not_change_the_answer():
    # The statement that nothing unrolls, read off the top. A one-element list
    # and an eight-element one are both a `cons`, and `sort` is told no more
    # than that about either.
    answers = set()

    for index, datum in enumerate(["(1)", "(3 1 2)", "(3 1 4 1 5 9 2 6)"]):
        forms, store = analysed(
            PASS + SORT + f"(sort (quote {datum}))", 2210 + index * 10
        )
        form, table = forms[-1]
        found = applied(form, table, store, procedures(forms, store))

        assert found is not None
        answers.add((found[0].identifier, found[1]))

    assert len(answers) == 1


def test_an_empty_literal_is_declined_for_want_of_an_element_type():
    # `'()` inhabits `(@list a)` for any `a`, and the checker has no instance
    # to read — so there is no type to split and no class to land in. Declined
    # rather than guessed, as `PatternCoverage.coverage` declines a scrutinee
    # whose type cannot be read.
    forms, store = analysed(PASS + SORT + "(sort (quote ()))", 2240)
    form, table = forms[-1]

    assert applied(form, table, store, procedures(forms, store)) is None


def test_a_form_applying_nothing_known_is_declined():
    forms, store = analysed(PASS + "(quote (1 2))", 2250)
    form, table = forms[-1]

    assert applied(form, table, store, procedures(forms, store)) is None


# -- the precondition ---------------------------------------------------------


def test_an_incomplete_match_is_refused():
    # A class is the claim that every value written that way reaches one
    # clause. A `match` leaving a value unmatched has values that reach none,
    # so the claim would be about nothing.
    #
    # Read on a form the checker never saw, because the checker refuses an
    # incomplete `match` outright.
    program, table = program_of("(match xs ((cons (prod# x r)) x))", 2300)
    store: Context = Context.default()
    store.register(Symbol("xs"), LIST)

    with pytest.raises(CoverageError, match=r"leaves \(nil _\) unmatched"):
        applied(program, table, store, {})


def test_the_procedures_of_a_program_are_found_across_its_forms():
    forms, store = analysed(PASS + SORT + "(sort (quote (1)))", 2310)
    known = procedures(forms, store)

    assert {binder.identifier for binder in known} == {"pass", "sort"}
    assert [
        parameter.identifier for parameter in known[Symbol("pass")].parameters
    ] == ["xs"]


def test_reading_a_type_does_not_write_the_callers_store():
    forms, store = analysed(PASS, 2320)
    before = dict(store.ascriptions)

    summaries(forms, store)

    assert store.ascriptions == before
