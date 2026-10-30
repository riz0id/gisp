from collections.abc import Callable
from itertools import count

from Analysis.Classes import classes, decided, demands, splits
from Expander import expand, parse
from Reader import read_all
from Resolve import BindingTable, bindings, meaning
from Syntax import Expr, Scope, SList, Symbol, binder_name
from Syntax.Type import BOOL, Context, IntType, ProdType, Type, list_of
from TypeCheck import TypeChecker, check_toplevel

# A procedure's walks are determined by the ways its argument can be *written*:
# an argument of `(@list @int)` is `(nil _)` or `(cons _)`, and a procedure
# whose clauses look deeper splits deeper — no further.
#
# That is what replaces following a call. The walks are computed once per
# class, so a recursive call is resolved by asking which classes its argument
# falls in, and nothing is unrolled to find out.
#
# Counters from the 2400s, as `test_walks` takes the 2000s.

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

# A clause asking something no constructor split can answer: an element being
# exactly one. The refinement has to stop rather than chase the spine.
LITERAL: str = """
(define (ascribe# first (@fun (@prod (@list @int)) @int))
  (lambda ((ascribe# zs (@list @int)))
    (match zs
      ((cons (prod# 1 r)) 1)
      (other 0))))
"""


def counter(start: int) -> Callable[[], Scope]:
    uids = count(start)
    return lambda: Scope(next(uids))


def analysed(source: str, at: int) -> tuple[list[tuple[Expr, BindingTable]], Context]:
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


def declared(
    source: str, at: int, name: str
) -> tuple[Symbol, Expr, Type, Context, BindingTable]:
    # One declared procedure's parameter, body and parameter type, read off the
    # form the way the analysis reads it: the binder a `define` declares, and
    # the one-parameter `lambda` it is initialised with.
    forms, store = analysed(source, at)

    for form, table in forms:
        match form:
            case SList(
                [
                    Symbol() as head,
                    declared_as,
                    SList([Symbol() as made, SList([written]), body]),
                ]
            ) if (
                head.identifier == "define"
                and made.identifier == "lambda"
                and (bound := binder_name(declared_as)) is not None
                and bound.identifier == name
                and (parameter := binder_name(written)) is not None
            ):
                binder: Symbol = meaning(parameter, table)

                return binder, body, store[binder], store, table

    raise AssertionError(f"no one-parameter procedure named {name}")


def written_as(patterns: list[Expr]) -> set[str]:
    return {str(pattern) for pattern in patterns}


# -- what a type says on its own ------------------------------------------------


def test_a_list_splits_two_ways():
    # The whole of what the type says: an argument is a `nil` or a `cons`, and
    # nothing deeper is implied by the type alone.
    assert written_as(splits(LIST, Context.default())) == {"(cons _)", "(nil _)"}


def test_a_boolean_splits_two_ways():
    assert written_as(splits(BOOL, Context.default())) == {"(#t _)", "(#f _)"}


def test_a_product_splits_one_way():
    # One constructor, so a product pattern refutes nothing.
    pair = ProdType([IntType(), IntType()])

    assert written_as(splits(pair, Context.default())) == {"(prod# _ _)"}


def test_a_ground_type_does_not_split():
    # `@int` has a constructor per value and no split enumerates them.
    assert splits(IntType(), Context.default()) == []


# -- the classes are what the clauses demand ------------------------------------


def test_a_parameter_no_match_reads_splits_only_by_its_type():
    # `sort` never matches `xs` — it matches `(eq xs ys)` — so the type's own
    # split is as fine as its classes get.
    parameter, body, ty, store, table = declared(PASS + SORT, 2400, "sort")

    assert demands(parameter, body, table) == []
    assert written_as(classes(parameter, body, ty, store, table)) == {
        "(cons _)",
        "(nil _)",
    }


def test_a_parameter_its_clauses_look_into_splits_as_deep_as_they_look():
    # `pass` writes three clauses: the empty list, the one-element list, and
    # the rest. So its argument has three classes, refined exactly to the depth
    # those clauses reach and no further — the tail of the third stays a
    # wildcard, because no clause looks past it.
    parameter, body, ty, store, table = declared(PASS, 2500, "pass")

    assert written_as(classes(parameter, body, ty, store, table)) == {
        "(nil _)",
        "(cons (prod# _ (nil _)))",
        "(cons (prod# _ (cons _)))",
    }


def test_each_class_decides_the_clause_it_reaches():
    # No `MAYBE` survives the refinement: every value of a class reaches the
    # same clause, which is what makes a class a walk rather than a set of
    # them.
    parameter, body, ty, store, table = declared(PASS, 2510, "pass")
    [written] = demands(parameter, body, table)

    for klass in classes(parameter, body, ty, store, table):
        assert decided(written, klass, ty, store, table)


def test_the_type_split_alone_does_not_decide_those_clauses():
    # The reason the refinement is needed: `(cons _)` reaches both the
    # one-element clause and the rest, so it says nothing about which walk a
    # cons takes.
    parameter, body, ty, store, table = declared(PASS, 2520, "pass")
    [written] = demands(parameter, body, table)
    coarse = [
        klass for klass in splits(ty, store) if str(klass) == "(cons _)"
    ]

    assert coarse
    assert not decided(written, coarse[0], ty, store, table)


# -- where the refinement must stop ---------------------------------------------


def test_a_clause_a_split_cannot_answer_stops_the_refinement():
    # `(cons (prod# 1 r))` asks whether the element is one, and no constructor
    # split of `@int` answers that. The class is left undecided rather than
    # refined forever — chasing it would unfold the spine without end, which is
    # the thing this analysis does not do.
    parameter, body, ty, store, table = declared(LITERAL, 2530, "first")
    found = classes(parameter, body, ty, store, table)

    assert written_as(found) == {"(nil _)", "(cons (prod# _ _))"}


def test_the_undecided_class_is_reported_as_undecided():
    parameter, body, ty, store, table = declared(LITERAL, 2540, "first")
    [written] = demands(parameter, body, table)
    verdicts = {
        str(klass): decided(written, klass, ty, store, table)
        for klass in classes(parameter, body, ty, store, table)
    }

    # The empty list reaches the catch-all for certain; a cons with an unknown
    # element may reach either clause.
    assert verdicts == {"(nil _)": True, "(cons (prod# _ _))": False}


def test_the_classes_of_a_ground_parameter_are_the_one_that_matches_anything():
    store: Context = Context.default()
    table: BindingTable = BindingTable()
    found = classes(Symbol("n"), Symbol("n"), IntType(), store, table)

    assert written_as(found) == {"_"}
