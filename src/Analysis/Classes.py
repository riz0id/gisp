from Analysis import matches
from Analysis.PatternCoverage import clause_pattern, clauses
from Analysis.Shape import Shape, Verdict, admits, naming, shaped
from Resolve import BindingTable, meaning
from Syntax import Expr, SList, Symbol, binder_name
from Syntax.Type import Context, Type, constructors
from TypeCheck.Patterns import (
    WILDCARD,
    irrefutable,
    operand_types,
    pattern_head,
    pattern_operands,
)

__all__: list[str] = [
    "classes",
    "decided",
    "demands",
    "splits",
]


def splits(ty: Type, store: Context) -> list[Expr]:
    # One pattern per constructor of `ty`, each written with a wildcard per
    # operand — the one-level case split a type makes on its own.
    #
    # This is the whole of what a type says about how a value of it can be
    # written: an argument of `(@list @int)` is `(nil _)` or `(cons _)`, and
    # nothing deeper is implied by the type. Anything deeper has to be demanded
    # by a pattern some clause wrote, which is what `expanded` below reads.
    #
    # Empty where the sort has no closed set of constructors. `@int` has one
    # per value and no split enumerates them, so a parameter of `@int` has the
    # one class that matches everything.
    cons: dict[Symbol, Type] | None = constructors(ty, store)

    if cons is None:
        return []

    return [
        SList([constructor, *[WILDCARD] * len(operand_types(ty, payload, store))])
        for constructor, payload in cons.items()
    ]


def demands(parameter: Symbol, body: Expr, table: BindingTable) -> list[list[Expr]]:
    # The clause patterns of each `match` in `body` that scrutinises
    # `parameter`, one list per `match`.
    #
    # What the procedure actually asks of its argument. A parameter no `match`
    # reads demands nothing, so its classes are the type's own split and no
    # finer — which is why `sort`, whose body never matches `xs`, splits two
    # ways where `pass` splits three.
    rets: list[list[Expr]] = []

    for path in matches(body):
        try:
            form: Expr = path.follow(body)
        except ValueError:
            continue

        read: tuple[Expr, list[Expr]] | None = clauses(form)

        if read is None:
            continue

        scrutinee, written = read

        if isinstance(scrutinee, Symbol) and meaning(scrutinee, table) == parameter:
            rets.append([clause_pattern(clause) for clause in written])

    return rets


def decided(
    written: list[Expr],
    klass: Expr,
    ty: Type,
    store: Context,
    table: BindingTable,
) -> bool:
    # Whether every value of `klass` reaches the same clause of `written`.
    #
    # Read by taking the class as the argument it stands for and asking each
    # clause about it in order: a refuted clause is passed over, and the first
    # clause that certainly matches is the one every value of the class
    # reaches. A `MAYBE` means some values reach it and some do not, so the
    # class is too coarse to say which walk they take.
    shape: Shape = shaped(klass, ty, store, table, naming())

    for clause in written:
        verdict: Verdict = admits(clause, shape, table, store, naming()).verdict

        if verdict is Verdict.NO:
            continue

        return verdict is Verdict.YES

    # No clause reached at all, which coverage rules out. Answered as decided
    # rather than refined forever: `Analysis.analysable` is what refuses such a
    # `match`, and refining would not repair it.
    return True


def expanded(
    klass: Expr,
    patterns: list[Expr],
    ty: Type,
    store: Context,
    table: BindingTable,
) -> list[Expr]:
    # `klass` with one binder position replaced by the constructors that
    # position can be, or empty where no position should be expanded.
    #
    # A position is expanded only where some pattern *names a constructor*
    # there, and that restriction is what makes this terminate. Expanding
    # wherever a class had a binder would walk down a recursive type forever:
    # a clause written `(cons (prod# 1 _))` leaves the element undecided
    # whatever the list is refined to, and chasing that would unfold the spine
    # without end.
    #
    # So the refinement stops at the depth the clauses were written and no
    # deeper. That is the same argument that makes
    # `TypeCheck.Patterns.missing` terminate over a `@mu`, and here it is the
    # whole claim that nothing is unrolled.
    if binder_name(klass) is not None:
        if not any(pattern_head(pattern, table) is not None for pattern in patterns):
            # Every pattern leaves this position open too, or asks a question
            # no split answers — an integer literal against `@int`. Either way
            # there is nothing a constructor split would decide.
            return []

        return splits(ty, store)

    match klass:
        case SList([Symbol() as head, *operands]):
            named: Symbol = meaning(head, table)
            columns: list[Type] | None = (
                operand_types(ty, cons[named], store)
                if (cons := constructors(ty, store)) is not None and named in cons
                else None
            )

            if columns is None or len(columns) != len(operands):
                return []

            # The patterns that can still match here, read as `specialized`
            # reads them: one naming another constructor cannot, and an
            # irrefutable one matches whatever this is and demands nothing of
            # its operands.
            alive: list[Expr] = [
                pattern
                for pattern in patterns
                if not irrefutable(pattern)
                and pattern_head(pattern, table) == named
                and len(pattern_operands(pattern)) == len(columns)
            ]

            for index, column in enumerate(columns):
                grown: list[Expr] = expanded(
                    operands[index],
                    [pattern_operands(pattern)[index] for pattern in alive],
                    column,
                    store,
                    table,
                )

                if grown:
                    return [
                        SList(
                            [
                                head,
                                *operands[:index],
                                written,
                                *operands[index + 1 :],
                            ]
                        )
                        for written in grown
                    ]

            return []
        case _:
            return []


def classes(
    parameter: Symbol,
    body: Expr,
    ty: Type,
    store: Context,
    table: BindingTable,
) -> list[Expr]:
    # The ways `parameter` can be written, refined until each says which walk
    # the procedure takes.
    #
    # The type's own split to begin with, then refined by the clause patterns
    # of every `match` the body makes on this parameter. A class is finished
    # when every one of those `match`es is decided by it, or when no position
    # is left that a clause names a constructor at — the second being the
    # honest end for a clause that asks something a constructor split cannot
    # answer.
    #
    # This is what replaces following a call. A procedure's walks are computed
    # once per class, so a recursive call is resolved by asking which classes
    # its argument falls in — never by traversing the body again.
    demanded: list[list[Expr]] = demands(parameter, body, table)
    patterns: list[Expr] = [pattern for written in demanded for pattern in written]
    rets: list[Expr] = []
    pending: list[Expr] = list(reversed(splits(ty, store) or [WILDCARD]))

    while pending:
        klass: Expr = pending.pop()

        if all(decided(written, klass, ty, store, table) for written in demanded):
            rets.append(klass)
            continue

        grown: list[Expr] = expanded(klass, patterns, ty, store, table)

        if not grown:
            rets.append(klass)
            continue

        pending.extend(reversed(grown))

    return rets
