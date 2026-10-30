from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import Final

from Analysis.CExpr import (
    AppCExpr,
    CExpr,
    CharCExpr,
    ComparisonCExpr,
    ConjunctionCExpr,
    DisjunctionCExpr,
    EquivalenceCExpr,
    IntegerCExpr,
    NegationCExpr,
    VarCExpr,
)
from Analysis.Path import Path
from Analysis.PatternCoverage import (
    Coverage,
    CoverageError,
    clause_body,
    clause_pattern,
    clauses,
    coverage,
)
from Analysis.Support import Support
from Resolve import BindingTable, meaning
from Syntax import Expr, Integer, SList, Symbol, binder_name
from Syntax.Type import (
    BOOL,
    FALSE,
    TRUE,
    CharType,
    Context,
    FunType,
    IntType,
    Type,
)


class OccurrenceError(Exception):
    pass


class Region(Enum):
    EXPR = auto()
    PAIRS = auto()
    PAIR = auto()
    CLAUSE = auto()


@dataclass(frozen=True, slots=True)
class Guard:
    test: CExpr
    pattern: Expr
    consequent: bool


def occurs_in(var: Symbol, expr: Expr, table: BindingTable) -> bool:
    target: Symbol = meaning(var, table)
    worklist: list[Expr] = [expr]

    while len(worklist) > 0:
        current: Expr = worklist.pop()

        match current:
            case Symbol() as symbol:
                if meaning(symbol, table) == target:
                    return True
            case SList([Symbol() as head, SList(), body]) if (
                head.identifier == "lambda"
            ):
                worklist.append(body)
            case SList([Symbol() as head, SList(pairs), body]) if (
                head.identifier == "letrec"
            ):
                worklist.append(body)

                for pair in pairs:
                    match pair:
                        case SList([name, initializer]) if (
                            binder_name(name) is not None
                        ):
                            worklist.append(initializer)
                        case _:
                            worklist.append(pair)
            case SList([Symbol() as head, scrutinee, *clauses]) if (
                head.identifier == "match"
            ) and clauses:
                worklist.append(scrutinee)
                worklist += [clause_body(clause) for clause in clauses]
            case SList([Symbol() as head, _, init]) if head.identifier == "define":
                worklist.append(init)
            case SList([Symbol() as head, operand]) if (
                head.identifier == "forall"
            ) and (bound := binder_name(operand)) is not None:
                if meaning(bound, table) == target:
                    return True
            case SList(items):
                worklist += items
            case _:
                pass

    return False


def conditions(
    var: Symbol,
    program: Expr,
    table: BindingTable,
) -> list[CExpr]:
    rets: list[Expr] = []
    worklist: list[Expr] = [program]

    while len(worklist) > 0:
        expr: Expr = worklist.pop()

        match expr:
            case SList([Symbol() as head, scrutinee, *clauses]) if (
                head.identifier == "match"
            ) and clauses:
                if occurs_in(var=var, expr=scrutinee, table=table):
                    rets.append(scrutinee)

                worklist += reversed(
                    [scrutinee, *[clause_body(clause) for clause in clauses]]
                )
            case SList([Symbol() as head, SList(), body]) if (
                head.identifier == "lambda"
            ):
                worklist.append(body)
            case SList([Symbol() as head, SList(pairs), body]) if (
                head.identifier == "letrec"
            ):
                initializers: list[Expr] = []

                for pair in pairs:
                    match pair:
                        case SList([name, initializer]) if (
                            binder_name(name) is not None
                        ):
                            initializers.append(initializer)
                        case _:
                            initializers.append(pair)

                worklist += reversed([*initializers, body])
            case SList([Symbol() as head, _, init]) if head.identifier == "define":
                worklist.append(init)
            case SList([Symbol() as head, _]) if head.identifier == "forall":
                pass
            case SList(items):
                worklist += reversed(items)
            case _:
                pass

    return [CExpr.from_expr(expr) for expr in rets]


def occurrences(
    var: Symbol,
    program: Expr,
    table: BindingTable,
) -> list[Path]:
    target: Symbol = meaning(var, table)
    rets: list[Path] = []
    worklist: list[tuple[Path, Expr]] = [(Path(), program)]

    while len(worklist) > 0:
        path, expr = worklist.pop()

        match expr:
            case Symbol() as symbol:
                if meaning(symbol, table) == target:
                    rets.append(path)
            case SList([Symbol() as head, SList(), body]) if (
                head.identifier == "lambda"
            ):
                worklist.append((path.child(2), body))
            case SList([Symbol() as head, SList(pairs), body]) if (
                head.identifier == "letrec"
            ):
                children: list[tuple[Path, Expr]] = []

                for position, pair in enumerate(pairs):
                    match pair:
                        case SList([name, initializer]) if (
                            binder_name(name) is not None
                        ):
                            children.append(
                                (path.child(1).child(position).child(1), initializer)
                            )
                        case _:
                            children.append((path.child(1).child(position), pair))

                children.append((path.child(2), body))
                worklist += reversed(children)
            case SList([Symbol() as head, scrutinee, *clauses]) if (
                head.identifier == "match"
            ) and clauses:
                reached: list[tuple[Path, Expr]] = [(path.child(1), scrutinee)]

                for position, clause in enumerate(clauses):
                    match clause:
                        case SList([_, body]):
                            reached.append((path.child(2 + position).child(1), body))
                        case _:
                            reached.append((path.child(2 + position), clause))

                worklist += reversed(reached)
            case SList([Symbol() as head, _, init]) if head.identifier == "define":
                worklist.append((path.child(2), init))
            case SList([Symbol() as head, *_]) if head.identifier == "quote":
                pass
            case SList([Symbol() as head, operand]) if (
                head.identifier == "forall"
            ) and (bound := binder_name(operand)) is not None:
                if meaning(bound, table) == target:
                    rets.append(path)
            case SList(items):
                worklist += reversed(
                    [
                        (path.child(position), item)
                        for position, item in enumerate(items)
                    ]
                )
            case _:
                pass

    return rets


def arms(program: Expr) -> list[Path]:
    rets: list[Path] = []
    worklist: list[tuple[Path, Expr]] = [(Path(), program)]

    while len(worklist) > 0:
        path, expr = worklist.pop()

        match expr:
            case SList([Symbol() as head, scrutinee, *clauses]) if (
                head.identifier == "match"
            ) and clauses:
                reached: list[tuple[Path, Expr]] = [(path.child(1), scrutinee)]

                for position, clause in enumerate(clauses):
                    match clause:
                        case SList([_, body]):
                            arm: Path = path.child(2 + position).child(1)
                            rets.append(arm)
                            reached.append((arm, body))
                        case _:
                            reached.append((path.child(2 + position), clause))

                worklist += reversed(reached)
            case SList([Symbol() as head, SList(), body]) if (
                head.identifier == "lambda"
            ):
                worklist.append((path.child(2), body))
            case SList([Symbol() as head, SList(pairs), body]) if (
                head.identifier == "letrec"
            ):
                children: list[tuple[Path, Expr]] = []

                for position, pair in enumerate(pairs):
                    match pair:
                        case SList([name, initializer]) if (
                            binder_name(name) is not None
                        ):
                            children.append(
                                (path.child(1).child(position).child(1), initializer)
                            )
                        case _:
                            children.append((path.child(1).child(position), pair))

                children.append((path.child(2), body))
                worklist += reversed(children)
            case SList([Symbol() as head, _, init]) if head.identifier == "define":
                worklist.append((path.child(2), init))
            case SList([Symbol() as head, *_]) if head.identifier == "quote":
                pass
            case SList([Symbol() as head, _]) if head.identifier == "forall":
                pass
            case SList(items):
                worklist += reversed(
                    [
                        (path.child(position), item)
                        for position, item in enumerate(items)
                    ]
                )
            case _:
                pass

    return rets


def opaque(program: Expr, table: BindingTable) -> list[Symbol]:
    rets: list[Symbol] = []
    worklist: list[Expr] = [program]

    while len(worklist) > 0:
        expr: Expr = worklist.pop()

        match expr:
            case SList([Symbol() as head, operand]) if (
                head.identifier == "forall"
            ) and (bound := binder_name(operand)) is not None:
                binder: Symbol = meaning(bound, table)

                if binder not in rets:
                    rets.append(binder)
            case SList([Symbol() as head, *_]) if head.identifier == "quote":
                pass
            case SList(items):
                worklist += reversed(items)
            case _:
                pass

    return rets


def matches(program: Expr) -> list[Path]:
    # Every `match` form in the program, by path.
    #
    # Read off `arms` rather than walked again. `parse` gives every `match` at
    # least one pattern/body clause, so `arms` records at least one arm for
    # each one, and an arm is a clause's body — two steps below the form. The
    # skipping is `arms`' too, which is what leaves a `match` inside quoted
    # data alone: that one is data and not a form.
    rets: list[Path] = []

    for arm in arms(program):
        form: Path = arm[:-2]

        if form not in rets:
            rets.append(form)

    return rets


def coverages(
    program: Expr,
    table: BindingTable,
    store: Context,
) -> list[tuple[Path, Coverage | None]]:
    # What each `match` in the program covers, in the order the forms are
    # reached. A `None` is a `match` coverage has nothing to say about, which
    # `PatternCoverage.coverage` is where the reading of that lives.
    return [
        (path, coverage(path.follow(program), table, store))
        for path in matches(program)
    ]


def incomplete(
    program: Expr,
    table: BindingTable,
    store: Context,
) -> tuple[Path, Coverage] | None:
    # The first `match` in the program that leaves a value unmatched, or `None`
    # where none does. A `match` with no verdict is not one of them: coverage
    # declining to answer is not an answer of "incomplete".
    for path, verdict in coverages(program, table, store):
        if verdict is not None and not verdict.complete():
            return path, verdict

    return None


def analysable(program: Expr, table: BindingTable, store: Context) -> None:
    # The precondition the readings below are made under, refused where it
    # fails.
    #
    # A guard says that reaching a clause means the clauses before it failed
    # and that one matched, which describes the program only where some clause
    # always matches. A `match` that leaves a value unmatched falls off its last
    # clause and has no value at all, so a series of guards about it describes a
    # path the program never takes — and a refinement read off that series is
    # about nothing.
    #
    # The type checker refuses such a program too, which is why nothing
    # reaching `--analyse` can be one. This is the same precondition stated
    # where the analysis itself is entered, for the caller that comes to
    # `refinements` with a program the checker never saw.
    found: tuple[Path, Coverage] | None = incomplete(program, table, store)

    if found is not None:
        path, verdict = found

        raise CoverageError(f"the match at {path} leaves {verdict.witness} unmatched.")


def selections(path: Path, program: Expr) -> list[tuple[Expr, int]]:
    # The clauses a path selects, as ⟨`match` form, clause position⟩ and in the
    # order the path reaches them.
    #
    # Read by prefix rather than by walking the program, which is what makes
    # the two path shapes the analysis is called with one case: a clause's own
    # path ends at the clause, and an arm's path — what `arms` answers — ends
    # one step below it, in the body.
    rets: list[tuple[Expr, int]] = []

    for length in range(1, len(path) + 1):
        prefix: Path = path[:length]
        position: int = prefix[-1]

        if position < 2:
            continue

        try:
            form: Expr = prefix[:-1].follow(program)
        except ValueError:
            # A prefix that does not descend means no longer one does either.
            break

        read: tuple[Expr, list[Expr]] | None = clauses(form)

        if read is not None and position - 2 < len(read[1]):
            rets.append((form, position - 2))

    return rets


def path_conditions(
    path: Path,
    program: Expr,
    table: BindingTable,
) -> list[Guard]:
    rets: list[Guard] = []
    expr: Expr = program
    region: Region = Region.EXPR

    for step, position in enumerate(path):
        if not isinstance(expr, SList):
            raise OccurrenceError(f"step {step} of {path} descends into {expr}")

        items: list[Expr] = expr.items

        if not 0 <= position < len(items):
            raise OccurrenceError(f"step {step} of {path} has no child in {expr}")

        match region:
            case Region.PAIRS:
                region = Region.PAIR
            case Region.PAIR:
                match items:
                    case [name, _] if position == 0 and binder_name(name) is not None:
                        raise OccurrenceError(f"step {step} of {path} selects {name}")
                    case _:
                        region = Region.EXPR
            case Region.CLAUSE:
                match items:
                    case [pattern, _] if position == 0:
                        raise OccurrenceError(
                            f"step {step} of {path} selects {pattern}"
                        )
                    case _:
                        region = Region.EXPR
            case Region.EXPR:
                match items:
                    case [Symbol() as head, scrutinee, *clauses] if (
                        head.identifier == "match"
                    ) and clauses:
                        if position >= 2 and position - 2 < len(clauses):
                            test: CExpr = CExpr.from_expr(scrutinee, table)
                            selected: int = position - 2

                            for earlier in clauses[:selected]:
                                rets.append(
                                    Guard(
                                        test=test,
                                        pattern=clause_pattern(earlier),
                                        consequent=False,
                                    )
                                )

                            rets.append(
                                Guard(
                                    test=test,
                                    pattern=clause_pattern(clauses[selected]),
                                    consequent=True,
                                )
                            )
                            region = Region.CLAUSE
                    case [Symbol() as head, SList(), _] if head.identifier == "lambda":
                        if position == 1:
                            raise OccurrenceError(
                                f"step {step} of {path} selects a parameter list"
                            )
                    case [Symbol() as head, _, _] if head.identifier == "define":
                        if position == 1:
                            raise OccurrenceError(
                                f"step {step} of {path} selects a define binder"
                            )
                    case [Symbol() as head, SList(), _] if head.identifier == "letrec":
                        if position == 1:
                            region = Region.PAIRS
                    case [Symbol() as head, *_] if head.identifier == "quote":
                        if position > 0:
                            raise OccurrenceError(
                                f"step {step} of {path} selects quoted data"
                            )
                    case [Symbol() as head, _] if head.identifier == "forall":
                        raise OccurrenceError(
                            f"step {step} of {path} descends into a free variable"
                        )
                    case _:
                        pass

        expr = items[position]

    return rets


class Relation(StrEnum):
    EQ = "eq"
    LT = "lt"


def comparison(relation: Relation, polarity: bool, bound: int) -> Support:
    match relation:
        case Relation.LT:
            support: Support = Support.less_than(bound)
        case Relation.EQ:
            support = Support.point(bound)

    return support if polarity else support.complement()


GROUNDS: Final[tuple[Type, ...]] = (IntType(), CharType(), BOOL)


def ascription(operand: CExpr, context: Context, table: BindingTable) -> Type | None:
    match operand:
        case IntegerCExpr():
            return IntType()
        case CharCExpr():
            return CharType()
        case VarCExpr(symbol):
            binder: Symbol = meaning(symbol, table)

            return context[binder] if binder in context else None
        case (
            ComparisonCExpr()
            | ConjunctionCExpr()
            | DisjunctionCExpr()
            | EquivalenceCExpr()
            | NegationCExpr()
        ):
            return BOOL
        case AppCExpr(head=head):
            # An application, and so the return type of whatever is applied. A
            # construction is no longer this node, so there is no constructor to
            # exclude by name here: `(cons (prod# x xs))` is a `ConCExpr` and
            # falls to the last arm, which is the right answer — a list has no
            # ascription to read and nothing below asks for one.
            applied: Symbol = meaning(head, table)

            if applied not in context:
                return None

            match context[applied]:
                case FunType(return_type=return_type):
                    return return_type
                case _:
                    return None
        case _:
            return None


def disjoint(left: Type, right: Type) -> bool:
    return left in GROUNDS and right in GROUNDS and left != right


def disjoint_operands(
    left: CExpr,
    right: CExpr,
    context: Context,
    table: BindingTable,
) -> bool:
    lhs: Type | None = ascription(left, context, table)
    rhs: Type | None = ascription(right, context, table)

    return lhs is not None and rhs is not None and disjoint(lhs, rhs)


def satisfying(
    test: CExpr,
    target: Symbol,
    table: BindingTable,
    polarity: bool,
    context: Context,
) -> Support:
    match test:
        case ComparisonCExpr(VarCExpr(symbol), IntegerCExpr(value)) if meaning(
            symbol, table
        ) == meaning(target, table):
            return comparison(Relation.LT, polarity, int(value))
        case ComparisonCExpr(IntegerCExpr() as lit, VarCExpr() as ref) if meaning(
            ref.symbol, table
        ) == meaning(target, table):
            return satisfying(
                NegationCExpr(
                    DisjunctionCExpr(
                        ComparisonCExpr(ref, lit),
                        EquivalenceCExpr(ref, lit),
                    )
                ),
                target,
                table,
                polarity,
                context,
            )
        case EquivalenceCExpr(left, right) if disjoint_operands(
            left, right, context, table
        ):
            return Support.empty() if polarity else Support.integers()
        case EquivalenceCExpr(VarCExpr(symbol), IntegerCExpr(value)) | EquivalenceCExpr(
            IntegerCExpr(value), VarCExpr(symbol)
        ) if meaning(symbol, table) == meaning(target, table):
            return comparison(Relation.EQ, polarity, int(value))
        case NegationCExpr(operand):
            return satisfying(operand, target, table, not polarity, context)
        case ConjunctionCExpr(left, right):
            lefts: Support = satisfying(left, target, table, polarity, context)
            rights: Support = satisfying(right, target, table, polarity, context)

            return lefts.intersection(rights) if polarity else lefts.union(rights)
        case DisjunctionCExpr(left, right):
            lefts: Support = satisfying(left, target, table, polarity, context)
            rights: Support = satisfying(right, target, table, polarity, context)

            return lefts.union(rights) if polarity else lefts.intersection(rights)
        case AppCExpr(head=head, operands=[lhs, rhs]):
            # `le` read as the disjunction it is, so one reading of `lt` and
            # `eq` serves it too. The operands are already conditions —
            # `AppCExpr` converts them, where `ListCExpr` held them as syntax
            # and this arm had to convert them again.
            builtin: Symbol = meaning(head, table)

            if builtin.scopes or builtin.identifier != "le":
                return Support.integers()

            return satisfying(
                DisjunctionCExpr(
                    ComparisonCExpr(lhs, rhs),
                    EquivalenceCExpr(lhs, rhs),
                ),
                target,
                table,
                polarity,
                context,
            )
        case _:
            return Support.integers()


def matching(
    test: CExpr,
    pattern: Expr,
    target: Symbol,
    table: BindingTable,
    polarity: bool,
    context: Context,
) -> Support:
    match pattern:
        case SList([Symbol() as head, _]) if meaning(head, table) == TRUE:
            # The payload pattern is matched and passed over. Reaching the
            # clause is the whole of what the refinement is about, and a `#t`
            # clause is reached exactly when the scrutinee is true however the
            # empty product it carries was written.
            return satisfying(test, target, table, polarity, context)
        case SList([Symbol() as head, _]) if meaning(head, table) == FALSE:
            return satisfying(test, target, table, not polarity, context)
        case Integer() as value if isinstance(test, VarCExpr) and meaning(
            test.symbol, table
        ) == meaning(target, table):
            return comparison(Relation.EQ, polarity, int(value))
        case _:
            return Support.integers()


def contradictory(
    test: CExpr,
    pattern: Expr,
    consequent: bool,
    context: Context,
    table: BindingTable,
) -> bool:
    return matching(
        test, pattern, Symbol.toplevel(), table, consequent, context
    ).is_empty()


def refinements(
    var: Symbol,
    path: Path,
    program: Expr,
    table: BindingTable,
    context: Context,
) -> list[Support]:
    analysable(program, table, context)

    guards: list[Guard] = path_conditions(path, program, table)
    rets: list[Support] = [Support.integers()]

    for guard in guards:
        support: Support = matching(
            guard.test, guard.pattern, var, table, guard.consequent, context
        )
        rets.append(rets[-1].intersection(support))

    return rets


def unreachable(
    path: Path,
    program: Expr,
    table: BindingTable,
    context: Context,
) -> bool:
    analysable(program, table, context)

    guards: list[Guard] = path_conditions(path, program, table)

    if any(
        contradictory(guard.test, guard.pattern, guard.consequent, context, table)
        for guard in guards
    ):
        return True

    # The structural reason beside the contradictory one. A guard is about what
    # the scrutinee may be, and reads a clause as unreachable where no value
    # satisfies the series; this reads a clause as unreachable where its
    # pattern matches nothing the clauses before it did not — which is true of
    # the clause whatever any variable may be.
    for form, position in selections(path, program):
        verdict: Coverage | None = coverage(form, table, context)

        if verdict is not None and position in verdict.redundant:
            return True

    return False
