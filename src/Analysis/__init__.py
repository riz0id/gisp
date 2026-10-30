from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import Final

from Analysis.CExpr import (
    CExpr,
    CharCExpr,
    ComparisonCExpr,
    ConjunctionCExpr,
    DisjunctionCExpr,
    EquivalenceCExpr,
    IntegerCExpr,
    ListCExpr,
    NegationCExpr,
    VarCExpr,
)
from Analysis.Path import Path
from Analysis.Support import Support
from Resolve import BindingTable, meaning
from Syntax import Expr, Integer, SList, Symbol, binder_name
from Syntax.Type import (
    BOOL,
    CONSTRUCTORS,
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


def clause_pattern(clause: Expr) -> Expr:
    match clause:
        case SList([pattern, _]):
            return pattern
        case _:
            return clause


def clause_body(clause: Expr) -> Expr:
    match clause:
        case SList([_, body]):
            return body
        case _:
            return clause


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


def constructs(head: Symbol, table: BindingTable) -> bool:
    return meaning(head, table) in CONSTRUCTORS


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
        case ListCExpr([Symbol() as head, *_]) if not constructs(head, table):
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
        case ListCExpr([Symbol() as head, left_operand, right_operand]):
            builtin: Symbol = meaning(head, table)

            if builtin.scopes or builtin.identifier != "le":
                return Support.integers()

            lhs: CExpr = CExpr.from_expr(left_operand, table)
            rhs: CExpr = CExpr.from_expr(right_operand, table)

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
        case SList([Symbol() as head]) if meaning(head, table) == TRUE:
            return satisfying(test, target, table, polarity, context)
        case SList([Symbol() as head]) if meaning(head, table) == FALSE:
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
    guards: list[Guard] = path_conditions(path, program, table)

    return any(
        contradictory(guard.test, guard.pattern, guard.consequent, context, table)
        for guard in guards
    )
