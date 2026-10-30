from collections.abc import Callable
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Final

from Analysis import (
    CoverageError,
    Guard,
    Relation,
    analysable,
    clause_body,
    clause_pattern,
    comparison,
)
from Analysis.CExpr import CExpr
from Analysis.Classes import classes
from Analysis.Measure import Constraints, Edge, extensions
from Analysis.PatternCoverage import probing
from Analysis.Shape import (
    Admission,
    CallShape,
    ConShape,
    GroundShape,
    OpaqueShape,
    ProductShape,
    Shape,
    Verdict,
    admits,
    classified,
    naming,
    shaped,
    topped,
    unknown,
)
from Analysis.Support import Support
from Expander import KEYWORDS
from Resolve import BindingTable, meaning
from Syntax import Char, Expr, Integer, SList, Symbol, binder_name
from Syntax.Type import (
    BOOL,
    CONS,
    FALSE,
    NIL,
    PRODUCT,
    TRUE,
    CharType,
    Context,
    FunType,
    IntType,
    ProdType,
    Type,
)
from TypeCheck import TypeChecker, TypeCheckError

__all__: list[str] = [
    "Branch",
    "Call",
    "Procedure",
    "Summary",
    "applied",
    "procedures",
    "summaries",
    "summarise",
]


class Unreadable(Exception):
    # A form the walk analysis cannot read, which is not the same as a program
    # it refuses. Private: every caller of `walks_of` is answered `None`
    # instead, the way `PatternCoverage.coverage` declines a question rather
    # than raising about it.
    pass


@dataclass(frozen=True, slots=True)
class Procedure:
    # A top-level `define` whose initializer is a `lambda`, with the ways each
    # of its parameters can be written.
    #
    # `classes` is what replaces following a call. A procedure's walks are
    # computed once per class, so an application is resolved by asking which
    # classes its arguments fall in — never by traversing the body again.
    parameters: tuple[Symbol, ...]
    body: Expr
    table: BindingTable
    classes: tuple[tuple[Expr, ...], ...]


@dataclass(frozen=True, slots=True)
class Call:
    # An application of a known procedure, as the classes its arguments fall
    # in — one set per parameter.
    #
    # The transition a walk makes instead of entering the callee. The argument
    # `(cons (prod# y ys))` with `ys` unknown is refuted by the `nil` class and
    # admitted by both `cons` classes, so it falls in two, and the walk
    # branches into the callee's walks at each without any of them being
    # traversed here. A cycle among these is what recursion comes out as.
    procedure: Symbol
    classes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class Branch:
    # One way a procedure's body can go, for one class of its arguments.
    #
    # The guards are what `path_conditions` answers for a path, in the same
    # order and the same `Guard` — but within this body alone. A callee's
    # guards are not spliced in, because the callee is not entered; `calls`
    # names it instead.
    guards: tuple[Guard, ...]

    # What the branch committed its class's ground binders to, which is what
    # weighs it.
    constraints: Constraints

    # The shape the body answers with, which may hold `CallShape`s where it
    # answers with what a call returns.
    result: Shape

    # The calls the branch makes.
    calls: tuple[Call, ...]

    # Whether the branch was decided by something the constraints record. A
    # branch that was not has no probability to report — see `State.measured`.
    measured: bool


@dataclass(frozen=True, slots=True)
class Summary:
    # Every way a procedure's data can move, by the class its arguments are in.
    #
    # Finite, and the whole answer: one entry per tuple of parameter classes,
    # and within each the branches the body takes. Nothing here is a partial
    # answer to be continued — a recursive call is a `CallShape` in a result,
    # not a walk left unfinished, so there is nothing to bound and nothing to
    # truncate.
    procedure: Symbol
    parameters: tuple[Symbol, ...]
    classes: tuple[tuple[Expr, ...], ...]
    branches: dict[tuple[int, ...], tuple[Branch, ...]]


# Why a form is answered as unwalkable rather than walked.
#
# Each is a question the analysis declines: a form it cannot read, a type it
# cannot be told, and a store that cannot say what a variable stands for. None
# is a refusal of the program — `CoverageError` is the one of those, and it is
# raised rather than caught.
DECLINED: Final[tuple[type[Exception], ...]] = (
    Unreadable,
    TypeCheckError,
    ValueError,
)


@dataclass(frozen=True, slots=True)
class Reading:
    # The half of the walk that does not change as one is followed.
    store: Context
    checker: TypeChecker
    procedures: dict[Symbol, Procedure]
    fresh: Callable[[], Symbol]


@dataclass(frozen=True, slots=True)
class State:
    # The half that does.
    #
    # One body is walked at a time and never a callee's, so there is no frame
    # to push and nothing to restore — which is why there is no depth here to
    # bound. `table` and `bound` are the class's parameters; `guards`,
    # `constraints` and `calls` are what the walk has committed to so far.
    table: BindingTable
    bound: dict[Symbol, Shape] = field(default_factory=dict)
    guards: tuple[Guard, ...] = ()
    constraints: Constraints = field(default_factory=Constraints)
    calls: tuple[Call, ...] = ()

    # Whether every branch taken so far is one the constraints record.
    #
    # A comparison between two named unknowns is recorded and so weighs; a test
    # on what a *call* returns is not, because this reads a call by reference
    # and so knows nothing of its result. `sort`'s `(eq xs ys)` is the second
    # kind, and a branch marked here is reported with no probability rather
    # than with a wrong one.
    measured: bool = True


# One outcome: the state a subexpression left behind, and the shape it had.
type Outcome = tuple[State, Shape]


def procedures(
    forms: list[tuple[Expr, BindingTable]], store: Context
) -> dict[Symbol, Procedure]:
    # The top-level procedures of a program, by the binder each is declared at.
    #
    # Across forms, which is what makes the analysis interprocedural: a
    # `define` mints no scope, so one form's binder and a later form's
    # reference to it are the same symbol. That is the same reason the driver
    # rebuilds a table per form and carries only the store between them.
    rets: dict[Symbol, Procedure] = {}

    for form, table in forms:
        match form:
            case SList([Symbol() as head, name, initializer]) if (
                head.identifier == "define"
            ) and (bound := binder_name(name)) is not None:
                match initializer:
                    case SList([Symbol() as made, SList(parameters), body]) if (
                        made.identifier == "lambda"
                    ):
                        named: list[Symbol] = [
                            parameter
                            for written in parameters
                            if (parameter := binder_name(written)) is not None
                        ]

                        if len(named) != len(parameters):
                            continue

                        binders: list[Symbol] = [
                            meaning(parameter, table) for parameter in named
                        ]

                        if any(binder not in store for binder in binders):
                            # A parameter the checker never ascribed, so there
                            # is no type to split and no classes to walk under.
                            continue

                        rets[meaning(bound, table)] = Procedure(
                            tuple(binders),
                            body,
                            table,
                            tuple(
                                tuple(
                                    classes(binder, body, store[binder], store, table)
                                )
                                for binder in binders
                            ),
                        )

    return rets


def truth(which: bool) -> Shape:
    # A boolean shape, written as the sum constructor applied to the empty
    # product it carries — which is how a boolean is written everywhere else.
    return ConShape(TRUE if which else FALSE, ProductShape((), ProdType()), BOOL)


def ground_of(value: Integer | Char) -> Shape:
    # A literal, which is the one ground value that needs no name.
    return GroundShape(
        None,
        Support.point(value.value),
        IntType() if isinstance(value, Integer) else CharType(),
    )


def compared(left: Shape, right: Shape) -> bool | None:
    # Whether two shapes stand for the same value, where that is decided.
    #
    # Three-valued for the reason `admits` is: two shapes may be equal, be
    # different, or not say. Decided by name at a ground leaf — the *same*
    # unknown is equal to itself, and that is the whole reason a ground unknown
    # carries a name — and structurally above one.
    #
    # This is what lets `(eq xs ys)` be read where it can be: two shapes
    # naming the same unknowns are certainly equal. Where one side is a
    # `CallShape` it cannot be read, and the test is a branch whose two sides
    # this does not weigh.
    match left, right:
        case (
            ConShape(constructor=named, payload=payload),
            ConShape(constructor=other, payload=carried),
        ):
            return compared(payload, carried) if named == other else False
        case (ProductShape(fields=fields), ProductShape(fields=others)):
            if len(fields) != len(others):
                return False

            inner: list[bool | None] = [
                compared(field, other)
                for field, other in zip(fields, others, strict=True)
            ]

            if any(verdict is False for verdict in inner):
                return False

            return True if all(verdict is True for verdict in inner) else None
        case (
            GroundShape(variable=variable, support=support),
            GroundShape(variable=other, support=carried),
        ):
            if variable is not None and variable == other:
                return True

            if support.intersection(carried).is_empty():
                return False

            if support.cardinality() == 1 == carried.cardinality():
                return support == carried

            return None
        case (OpaqueShape() | CallShape(), _) | (_, OpaqueShape() | CallShape()):
            # ⊤, or what a call returns. Neither says anything, and a call's
            # result in particular must not read as *different*: this knows
            # nothing of it, which is not the same as knowing it is not equal.
            return None
        case _:
            # A construction of one sort against a construction of another, or
            # a construction against a ground: different however they are read.
            return False


def edged(builtin: str, lesser: Symbol, greater: Symbol, taken: bool) -> Edge:
    # The order a comparison commits to, each way.
    #
    # `le` taken says the left is not above the right, and `le` failed says the
    # right is strictly below the left. `lt` is the same two facts with the
    # strictness moved, which is what makes the two branches of one comparison
    # exclusive and exhaustive — and that is what makes the walks of a form
    # partition its inputs rather than merely cover them.
    if builtin == "le":
        return Edge(lesser, greater, False) if taken else Edge(greater, lesser, True)

    return Edge(lesser, greater, True) if taken else Edge(greater, lesser, False)


def point_of(support: Support) -> int | None:
    # The one value a support holds, or `None` where it holds more than one.
    match support.intervals:
        case ((value, same),) if value == same:
            return value
        case _:
            return None


def branched(state: State, constraints: Constraints, taken: bool) -> Outcome:
    return (replace(state, constraints=constraints), truth(taken))


def unweighed(state: State, taken: bool) -> Outcome:
    # A branch on something no constraint records, so the walk takes it but
    # cannot say how often.
    return (replace(state, measured=False), truth(taken))


def comparing(builtin: str, left: Shape, right: Shape, state: State) -> list[Outcome]:
    # `le` or `lt` over two shapes, read as the fact each way commits to.
    #
    # Four readings, by how much the two sides say. Two known values decide the
    # comparison and it is not a branch at all; two unknowns make it an order
    # between them; one of each makes it a bound on the one; and anything else
    # is a branch that commits to nothing.
    match left, right:
        case (
            GroundShape(variable=variable, support=support),
            GroundShape(variable=other, support=carried),
        ):
            pass
        case _:
            # Not two grounds — one side is a call's result, or an arithmetic
            # value — so nothing is known and both ways are open, with neither
            # recorded.
            return [unweighed(state, True), unweighed(state, False)]

    low: int | None = point_of(support)
    high: int | None = point_of(carried)

    if low is not None and high is not None:
        return [(state, truth(low < high if builtin == "lt" else low <= high))]

    if variable is not None and other is not None:
        return [
            branched(
                state,
                state.constraints.ordering(edged(builtin, variable, other, taken)),
                taken,
            )
            for taken in (True, False)
        ]

    # One side is a known value, so the comparison is a bound on the other.
    # `Analysis.comparison` is what reads a bound into a support, so this
    # narrows to what a guard on the same comparison narrows to.
    if variable is not None and high is not None:
        # `x ⊙ k` holds exactly below a limit, so the branch taken is the
        # support below it and the branch failed is the rest.
        limit: int = high if builtin == "lt" else high + 1

        return [
            branched(
                state,
                state.constraints.narrowing(
                    variable, comparison(Relation.LT, taken, limit)
                ),
                taken,
            )
            for taken in (True, False)
        ]

    if other is not None and low is not None:
        # `k ⊙ x` holds exactly *above* a limit, so the two branches trade
        # places.
        limit = low + 1 if builtin == "lt" else low

        return [
            branched(
                state,
                state.constraints.narrowing(
                    other, comparison(Relation.LT, not taken, limit)
                ),
                taken,
            )
            for taken in (True, False)
        ]

    # An unnamed value that is not a known one — what arithmetic answers with —
    # so there is nothing to record on either branch.
    return [unweighed(state, True), unweighed(state, False)]


def equalities(left: Shape, right: Shape) -> list[tuple[Symbol, Symbol]] | None:
    # The ground unknowns that must be equal for two shapes to be, or `None`
    # where the two cannot be lined up at all.
    #
    # Equality of two constructions is the conjunction of equality of their
    # leaves, which is what lets `(eq xs ys)` over two lists be weighed rather
    # than merely branched on. Without it the taken branch would commit to
    # nothing and so admit every ordering, and the branches of one test would
    # sum to more than the inputs they divide.
    match left, right:
        case (
            ConShape(constructor=named, payload=payload),
            ConShape(constructor=other, payload=carried),
        ) if named == other:
            return equalities(payload, carried)
        case (ProductShape(fields=fields), ProductShape(fields=others)) if len(
            fields
        ) == len(others):
            rets: list[tuple[Symbol, Symbol]] = []

            for field, other in zip(fields, others, strict=True):
                inner: list[tuple[Symbol, Symbol]] | None = equalities(field, other)

                if inner is None:
                    return None

                rets.extend(inner)

            return rets
        case (
            GroundShape(variable=variable),
            GroundShape(variable=other),
        ) if variable is not None and other is not None:
            return [] if variable == other else [(variable, other)]
        case _:
            return None


def equating(left: Shape, right: Shape, state: State) -> list[Outcome]:
    # `eq` over two shapes.
    #
    # Where equality is decided there is one outcome, which is what stops
    # bubblesort's `sort`. Where it is not, two — and over two ground unknowns
    # the taken branch records that each is below the other, which no ordering
    # of distinct elements satisfies. So `extensions` weighs that branch zero
    # without this having to assume anything about it.
    decided: bool | None = compared(left, right)

    if decided is not None:
        return [(state, truth(decided))]

    pairs: list[tuple[Symbol, Symbol]] | None = equalities(left, right)

    if pairs is None:
        # Two shapes that cannot be lined up — `sort` comparing its argument
        # with what `pass` returns is the case — so neither way says anything
        # and neither is weighed.
        return [unweighed(state, True), unweighed(state, False)]

    constraints: Constraints = state.constraints

    for variable, other in pairs:
        # Each below the other, which is how equality is written in an
        # ordering. No ordering of distinct keys satisfies it, so this branch
        # weighs nothing — `extensions` finds the cycle and counts zero,
        # without this having to assume anything about it.
        constraints = constraints.ordering(Edge(variable, other, False)).ordering(
            Edge(other, variable, False)
        )

    return [branched(state, constraints, True), (state, truth(False))]


ARITHMETIC: Final[frozenset[str]] = frozenset({"+", "-", "*", "mod"})


def arithmetic(builtin: str, shapes: list[Shape], state: State) -> list[Outcome]:
    # `+`, `-`, `*` and `mod` over shapes.
    #
    # Exact on two literals and ⊤ otherwise, which is as far as a `Support` of
    # disjoint intervals goes: the sum of two intervals is an interval, but the
    # product of two is not, and widening one sort of answer and not the other
    # would make the arithmetic read differently depending on the operator.
    #
    # Total where the evaluator refuses. `mod` by zero raises there, and here it
    # is a walk whose result says nothing — which is what `Evaluate.Builtins`
    # already anticipates in saying that only the stochastic analysis is total.
    match shapes:
        case [GroundShape(support=left), GroundShape(support=right)] if (
            value := point_of(left)
        ) is not None and (other := point_of(right)) is not None:
            match builtin:
                case "+":
                    return [(state, ground_of(Integer(value + other)))]
                case "-":
                    return [(state, ground_of(Integer(value - other)))]
                case "*":
                    return [(state, ground_of(Integer(value * other)))]
                case _ if other != 0:
                    return [(state, ground_of(Integer(value % other)))]

    return [(state, GroundShape(None, Support.integers(), IntType()))]


def selecting(builtin: str, shapes: list[Shape], state: State) -> list[Outcome]:
    # `head`, `tail` and `len` over shapes.
    #
    # Each reads a cons cell, and a cell is a `ConShape` over the product of an
    # element and a rest — the shape `Syntax.Expr.Con.cell` reads over values.
    # Nothing is refused where the shape does not say: an empty list is an
    # error the evaluator raises and a walk that says nothing here.
    match shapes, builtin:
        case [ConShape(constructor=named, payload=ProductShape(fields=fields))], _ if (
            named == CONS
        ) and len(fields) == 2:
            element, rest = fields

            match builtin:
                case "head":
                    return [(state, element)]
                case "tail":
                    return [(state, rest)]
        case [ConShape(constructor=named)], "len" if named == NIL:
            return [(state, ground_of(Integer(0)))]

    if builtin == "len":
        return [(state, GroundShape(None, Support.integers(), IntType()))]

    raise Unreadable(f"{builtin} over a shape that is not a cell")


def constructing(binder: Symbol, shapes: list[Shape]) -> Shape | None:
    # The shape a construction builds, or `None` where the head constructs
    # nothing.
    #
    # Built with no type, for the reason `Shape.ty` is allowed to be absent: a
    # construction is checked against the type its position wants and carries
    # no ascription to synthesize one from. A boolean is the exception only
    # because `BOOL` is what `#t` and `#f` always build.
    if binder == PRODUCT:
        return ProductShape(tuple(shapes))

    match shapes:
        case [payload] if binder in (TRUE, FALSE):
            return ConShape(binder, payload, BOOL)
        case [payload] if binder in (CONS, NIL):
            return ConShape(binder, payload)
        case _:
            return None


def returns(binder: Symbol, reading: Reading) -> Type:
    # What a procedure answers with, which is the type a `CallShape` standing
    # for one of its results is at.
    match reading.store[binder] if binder in reading.store else None:
        case FunType(return_type=return_type):
            return return_type
        case _:
            raise Unreadable(f"{binder} is not ascribed a function type")


def referred(
    binder: Symbol,
    procedure: Procedure,
    shapes: list[Shape],
    state: State,
    reading: Reading,
) -> Outcome:
    # One call, as the classes its arguments fall in.
    #
    # Where the old reading entered the callee's body and counted how deep it
    # had gone, this asks a question of the argument and stops: which of the
    # callee's classes can it be? The answer is a `CallShape` for the result
    # and a `Call` recorded on the walk, so the callee's own walks are joined
    # to this one by reference.
    #
    # This is the whole of why no budget is needed. A recursive call resolves
    # to a class of the procedure already being summarised, which is a cycle in
    # the summary rather than a recursion in the analysis.
    if len(procedure.parameters) != len(shapes):
        raise Unreadable(f"{binder} applied to {len(shapes)} arguments")

    indices: tuple[tuple[int, ...], ...] = tuple(
        classified(shape, written, reading.store, procedure.table, reading.fresh)
        for shape, written in zip(shapes, procedure.classes, strict=True)
    )

    return (
        replace(state, calls=(*state.calls, Call(binder, indices))),
        CallShape(binder, indices, returns(binder, reading)),
    )


def called(
    head: Expr, shapes: list[Shape], state: State, reading: Reading
) -> list[Outcome]:
    # A head applied to shapes, read as the first of the three things it can
    # be: a construction, a procedure, or a builtin.
    if not isinstance(head, Symbol):
        raise Unreadable(f"an application head that is not a symbol: {head}")

    binder: Symbol = meaning(head, state.table)
    built: Shape | None = None if binder.scopes else constructing(binder, shapes)

    if built is not None:
        return [(state, built)]

    if binder in reading.procedures:
        return [referred(binder, reading.procedures[binder], shapes, state, reading)]

    if not binder.scopes:
        match binder.identifier, shapes:
            case ("le" | "lt") as builtin, [left, right]:
                # The two comparisons are where an ordering is learned, so they
                # are where a walk can have just become one no input takes.
                return feasible(comparing(builtin, left, right, state))
            case "eq", [left, right]:
                return feasible(equating(left, right, state))
            case builtin, [_, _] if builtin in ARITHMETIC:
                return arithmetic(builtin, shapes, state)
            case ("head" | "tail" | "len") as builtin, [_]:
                return selecting(builtin, shapes, state)

    # Applied to something this cannot follow — a parameter holding a
    # procedure, or a builtin with no reading here — so the result is ⊤ at
    # whatever the head answers with.
    return [(state, unknown(returns(binder, reading), reading.store, reading.fresh))]


def sequenced(
    exprs: list[Expr], state: State, reading: Reading
) -> list[tuple[State, list[Shape]]]:
    # Several subexpressions in order, as every way they can jointly turn out.
    #
    # The product `arms` never forms. Two sibling `match`es in one application
    # are two independent case splits, and the ways their application can go is
    # the product of theirs — where `arms` answers one clause position per
    # clause and so counts them added rather than multiplied.
    rets: list[tuple[State, list[Shape]]] = [(state, [])]

    for expr in exprs:
        grown: list[tuple[State, list[Shape]]] = []

        for after, shapes in rets:
            grown.extend(
                (later, [*shapes, shape])
                for later, shape in walks(expr, after, reading)
            )

        rets = grown

    return rets


def branching(
    scrutinee: Expr, written: list[Expr], state: State, reading: Reading
) -> list[Outcome]:
    # A `match`, fanned out over the clauses its scrutinee's shape admits.
    #
    # The step `Evaluate.evaluate` cannot take. There, a `match` on an unknown
    # scrutinee rebuilds itself and returns unreduced, because an unknown value
    # selects no clause. Here the shape knows its type, so `admits` can answer
    # `MAYBE` — and a `MAYBE` is a branch.
    #
    # Coverage is what makes this total. Some clause of a complete `match`
    # admits every value of the scrutinee's type, so a shape that admits no
    # clause at all cannot arise from a program `analysable` passed; a shape
    # that admits one *certainly* reaches no later clause, which is why the loop
    # stops there.
    rets: list[Outcome] = []

    for after, decided in walks(scrutinee, state, reading):
        test: CExpr = CExpr.from_expr(scrutinee, after.table)
        refers: Symbol | None = (
            meaning(scrutinee, after.table) if isinstance(scrutinee, Symbol) else None
        )
        failed: list[Expr] = []

        for clause in written:
            pattern: Expr = clause_pattern(clause)
            verdict: Admission = admits(
                pattern, decided, after.table, reading.store, reading.fresh
            )
            failed.append(pattern)

            if verdict.verdict is Verdict.NO:
                # No value of this shape reaches the clause, so it is not even a
                # branch. Its failure still stands among the guards below, as
                # `path_conditions` records every clause tried before the one
                # taken.
                continue

            constraints: Constraints = after.constraints

            for variable, support in verdict.narrowed.items():
                constraints = constraints.narrowing(variable, support)

            bound: dict[Symbol, Shape] = {**after.bound, **verdict.bound}

            if refers is not None:
                # The scrutinee itself, narrowed by what the clause committed
                # it to: `((nil (prod#)) xs)` answers with the very list it has
                # just learned is empty.
                bound[refers] = verdict.refined

            rets.extend(
                walks(
                    clause_body(clause),
                    State(
                        table=after.table,
                        bound=bound,
                        guards=(
                            *after.guards,
                            *(
                                Guard(test=test, pattern=earlier, consequent=False)
                                for earlier in failed[:-1]
                            ),
                            Guard(test=test, pattern=pattern, consequent=True),
                        ),
                        constraints=constraints,
                        calls=after.calls,
                        measured=after.measured,
                    ),
                    reading,
                )
            )

            if verdict.verdict is Verdict.YES:
                # Certainly matched, so no later clause is reached by any value
                # of this shape.
                break

    return rets


def bound_shape(symbol: Symbol, state: State, reading: Reading) -> Shape:
    binder: Symbol = meaning(symbol, state.table)

    if binder in state.bound:
        return state.bound[binder]

    if binder in reading.store:
        # A binder the store ascribes but the walk never bound: a top-level
        # name, or a free variable. ⊤ at its type, which is all a reference can
        # say.
        return unknown(reading.store[binder], reading.store, reading.fresh)

    raise Unreadable(f"no shape for {symbol}")


def typed(expr: Expr, state: State, reading: Reading) -> Type:
    # The type a form already has, read through the checker so the analysis
    # does not keep a second opinion about one.
    checker: TypeChecker = replace(reading.checker, table=state.table)

    try:
        return checker.type_of(expr)
    except TypeCheckError as error:
        raise Unreadable(str(error)) from error


def feasible(outcomes: list[Outcome]) -> list[Outcome]:
    # `outcomes` with the ones no input satisfies dropped.
    #
    # An infeasible walk is satisfied by no value of the class at all, so
    # dropping it changes no weight and loses no answer.
    #
    # Read where a constraint is *added* and not at every node, because the
    # test closes the ordering transitively and a walk that learned nothing
    # cannot have become infeasible since it was last asked.
    return [outcome for outcome in outcomes if not outcome[0].constraints.infeasible()]


def walks(expr: Expr, state: State, reading: Reading) -> list[Outcome]:
    # Every way an expression can turn out, as the state each leaves and the
    # shape it has.
    #
    # `Evaluate.evaluate` arm for arm, answering a list where that answers a
    # value. A form with no `match` and no comparison in it has exactly one
    # outcome; every branch a shape cannot decide multiplies the outcomes. The
    # recursion is over the expression and nothing else — an application does
    # not descend into what it applies — so this terminates on the body's size
    # and needs no bound of its own.
    match expr:
        case Integer() | Char() as value:
            return [(state, ground_of(value))]
        case Symbol() as symbol:
            return [(state, bound_shape(symbol, state, reading))]
        case SList([]):
            raise Unreadable("the empty combination")
        case SList([Symbol() as head, datum]) if head.identifier == "quote":
            # The one place a datum becomes a shape, as it is the one place a
            # datum becomes a value — and read one level, so a literal's length
            # is never what an answer depends on.
            return [
                (
                    state,
                    topped(
                        datum,
                        typed(expr, state, reading),
                        reading.store,
                        reading.fresh,
                    ),
                )
            ]
        case SList([Symbol() as head, _]) if head.identifier == "forall":
            return [
                (
                    state,
                    unknown(typed(expr, state, reading), reading.store, reading.fresh),
                )
            ]
        case SList([Symbol() as head, scrutinee, *written]) if (
            head.identifier == "match"
        ) and written:
            return branching(scrutinee, written, state, reading)
        case SList([Symbol() as head, SList(pairs), body]) if (
            head.identifier == "letrec"
        ):
            return letting(pairs, body, state, reading)
        case SList([Symbol() as head, *_]) if head.identifier == "lambda":
            # A procedure is not data this reads, and only a top-level one is
            # followed into. An inner `lambda` is where a walk stops.
            raise Unreadable(f"an inner lambda: {expr}")
        case SList([Symbol() as head, *_]) if head.identifier == "define":
            raise Unreadable("define is not an expression")
        case SList(items):
            return [
                outcome
                for after, shapes in sequenced(items[1:], state, reading)
                for outcome in called(items[0], shapes, after, reading)
            ]
        case _:
            raise Unreadable(f"not an expression: {expr}")


def letting(
    pairs: list[Expr], body: Expr, state: State, reading: Reading
) -> list[Outcome]:
    # A `letrec`, with each initializer walked and bound before the body.
    #
    # Read in order and not as a fixpoint. A binder is visible in its own
    # initializer and a shape for it is what a fixpoint would compute, so one
    # that refers to itself is ⊤ here — which is sound and is where this stops
    # short of the evaluator. Bubblesort's `sort` writes a `letrec` that does
    # not: `ys` is `(pass xs)` and reaches nothing of itself.
    initializers: list[tuple[Symbol, Expr]] = []

    for pair in pairs:
        match pair:
            case SList([name, initializer]) if (bound := binder_name(name)) is not None:
                initializers.append((meaning(bound, state.table), initializer))
            case _:
                raise Unreadable(f"not a letrec binding: {pair}")

    rets: list[Outcome] = [(state, ProductShape(()))]

    for binder, initializer in initializers:
        grown: list[Outcome] = []

        for after, _ in rets:
            for later, shape in walks(initializer, after, reading):
                grown.append(
                    (replace(later, bound={**later.bound, binder: shape}), shape)
                )

        rets = grown

    return [outcome for after, _ in rets for outcome in walks(body, after, reading)]


def elements(branches: tuple[Branch, ...]) -> tuple[Symbol, ...]:
    # The ground unknowns a class's branches speak of.
    #
    # The union across the class and not one branch's own, because a weight is
    # a share: the branches of one class divide the same values between them,
    # so they have to be counted over the same set or their shares would not
    # sum to it.
    rets: list[Symbol] = []

    for branch in branches:
        for variable in sorted(branch.constraints.variables(), key=str):
            if variable not in rets:
                rets.append(variable)

    return tuple(rets)


def weight(branch: Branch, over: tuple[Symbol, ...]) -> int | None:
    # How many orderings of `over` take this branch, or `None` where the branch
    # was not decided by anything an ordering says.
    if not branch.measured:
        return None

    return extensions(branch.constraints, over)


def summarise(
    binder: Symbol,
    known: dict[Symbol, Procedure],
    store: Context,
) -> Summary | None:
    # Every way one procedure's data can move, by the class its arguments are
    # in — or `None` where its body is not one this can read.
    #
    # The body is walked once per tuple of parameter classes, and that is the
    # whole of the work: finitely many classes, each walked once, with a call
    # answered by reference. Nothing here is iterated to a fixpoint and nothing
    # is cut short, so there is no bound to choose and none to report.
    #
    # `analysable` first, which is the precondition and not a courtesy. A class
    # is the claim that every value written that way reaches one clause, and a
    # `match` leaving a value unmatched has values that reach none.
    procedure: Procedure = known[binder]

    try:
        analysable(procedure.body, procedure.table, store)
    except CoverageError:
        raise
    except DECLINED:
        return None

    branches: dict[tuple[int, ...], tuple[Branch, ...]] = {}

    for indices in product(*(range(len(written)) for written in procedure.classes)):
        fresh: Callable[[], Symbol] = naming()
        reading: Reading = Reading(
            store=store,
            # On a copied store, as `PatternCoverage.coverage` reads a type on
            # one: `type_of` registers the binders it reads through, and the
            # analysis is asking a question rather than checking a program.
            checker=TypeChecker(table=procedure.table, store=probing(store)),
            procedures=known,
            fresh=fresh,
        )
        bound: dict[Symbol, Shape] = {
            parameter: shaped(
                procedure.classes[position][index],
                store[parameter],
                store,
                procedure.table,
                fresh,
            )
            for position, (parameter, index) in enumerate(
                zip(procedure.parameters, indices, strict=True)
            )
        }

        try:
            outcomes: list[Outcome] = walks(
                procedure.body, State(procedure.table, bound), reading
            )
        except DECLINED:
            return None

        branches[indices] = tuple(
            Branch(
                guards=after.guards,
                constraints=after.constraints,
                result=shape,
                calls=after.calls,
                measured=after.measured,
            )
            for after, shape in outcomes
        )

    return Summary(
        procedure=binder,
        parameters=procedure.parameters,
        classes=procedure.classes,
        branches=branches,
    )


def summaries(
    forms: list[tuple[Expr, BindingTable]], store: Context
) -> dict[Symbol, Summary | None]:
    # Every top-level procedure of a program, summarised.
    known: dict[Symbol, Procedure] = procedures(forms, store)

    return {binder: summarise(binder, known, store) for binder in known}


def applied(
    form: Expr,
    table: BindingTable,
    store: Context,
    known: dict[Symbol, Procedure],
) -> tuple[Symbol, tuple[tuple[int, ...], ...]] | None:
    # The procedure a form applies and the classes its arguments fall in, or
    # `None` where the form applies no known procedure.
    #
    # What a report says about an expression: not a walk set of its own, but
    # which of the callee's classes it lands in. `(sort '(3 1 4 1 5 9 2 6))` is
    # `sort` applied to a `cons`, which is all the literal says.
    analysable(form, table, store)

    match form:
        case SList([Symbol() as head, *operands]) if (
            operands and head.identifier not in KEYWORDS
        ):
            pass
        case _:
            return None

    binder: Symbol = meaning(head, table)

    if binder not in known:
        return None

    fresh: Callable[[], Symbol] = naming()
    reading: Reading = Reading(
        store=store,
        checker=TypeChecker(table=table, store=probing(store)),
        procedures=known,
        fresh=fresh,
    )

    try:
        staged: list[tuple[State, list[Shape]]] = sequenced(
            operands, State(table), reading
        )
    except DECLINED:
        return None

    if not staged:
        return None

    procedure: Procedure = known[binder]
    _, shapes = staged[0]

    if len(shapes) != len(procedure.parameters):
        return None

    return binder, tuple(
        classified(shape, written, store, procedure.table, fresh)
        for shape, written in zip(shapes, procedure.classes, strict=True)
    )
