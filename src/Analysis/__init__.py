from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import Final

from Analysis.CExpr import (
    BooleanCExpr,
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
from Syntax import Expr, SList, Symbol, binder_name
from Syntax.Type import (
    CONSTRUCTORS,
    BoolType,
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


@dataclass(frozen=True, slots=True)
class Guard:
    test: CExpr
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
            case SList([Symbol() as head, _, init]) if head.identifier == "define":
                # Position 1 is a binding site, not an occurrence, so only the
                # initializer is searched.
                worklist.append(init)
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
            case SList([Symbol() as head, test, consequent, alternative]) if (
                head.identifier == "if"
            ):
                if occurs_in(var=var, expr=test, table=table):
                    rets.append(test)

                worklist += reversed([test, consequent, alternative])
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
            case SList([Symbol() as head, _, init]) if head.identifier == "define":
                # Position 1 is a binding site, not an occurrence, so only the
                # initializer is enqueued.
                worklist.append((path.child(2), init))
            case SList([Symbol() as head, *_]) if head.identifier == "quote":
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
            case Region.EXPR:
                match items:
                    case [Symbol() as head, test, _, _] if head.identifier == "if":
                        if position in (2, 3):
                            rets.append(
                                Guard(
                                    test=CExpr.from_expr(test, table),
                                    consequent=position == 2,
                                )
                            )
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


# The types whose values have a representation of their own, and which are
# therefore pairwise disjoint: an `Integer` is never a `Char`, and neither is
# ever one of the symbols `#t` and `#f` a comparison answers with.
GROUNDS: Final[tuple[Type, ...]] = (IntType(), CharType(), BoolType())


def constructs(head: Symbol, table: BindingTable) -> bool:
    # Whether a head names a constructor rather than an application head.
    # `check` tells the two apart by the type it is checking against, which a
    # shallow reading does not have, so a constructor is left unread:
    # `Context.default()` ascribes `prod#` a function returning the empty
    # product, and `(prod# 1 #t)` is a two-field product and not that.
    #
    # The identifiers come from `Syntax.Type`, where `is_constructor` compares a
    # resolved head against the one its sort expects.
    return meaning(head, table) in CONSTRUCTORS


def ascription(operand: CExpr, context: Context, table: BindingTable) -> Type | None:
    # The type of an operand when the ascriptions determine one, and `None`
    # otherwise. Deliberately shallow: the type checker only checks, so there
    # is nothing to ask for a type, and an operand the analysis cannot type is
    # one it declines to read.
    match operand:
        case IntegerCExpr():
            return IntType()
        case CharCExpr():
            return CharType()
        case BooleanCExpr():
            return BoolType()
        case VarCExpr(symbol):
            # The arm that types `#t` and `#f`: both read as ordinary symbols,
            # and `Context.default()` ascribes them `@bool`.
            binder: Symbol = meaning(symbol, table)

            return context[binder] if binder in context else None
        case (
            ComparisonCExpr()
            | ConjunctionCExpr()
            | DisjunctionCExpr()
            | EquivalenceCExpr()
            | NegationCExpr()
        ):
            # `lt`, `and`, `or`, `eq`, and `not` all answer with a boolean.
            return BoolType()
        case ListCExpr([Symbol() as head, *_]) if not constructs(head, table):
            applied: Symbol = meaning(head, table)

            if applied not in context:
                return None

            # An application has the return type of whatever its head is, and
            # a head ascribed anything but a function type says nothing.
            match context[applied]:
                case FunType(return_type=return_type):
                    return return_type
                case _:
                    return None
        case _:
            # A headless list, a construction, and `NilCExpr` — of which only a
            # construction is reachable from source.
            return None


def disjoint(left: Type, right: Type) -> bool:
    # Whether no value inhabits both types, which is what licenses reading an
    # `eq` between them as false in every state: `eq` is total over `Expr`, and
    # every sort's `__eq__` answers `False` for a value of another sort.
    #
    # Only the ground types qualify, and every exclusion is load-bearing. The
    # empty list inhabits `(@list t)` for every `t`, so two list types are not
    # disjoint; lists, products, and sums are all `SList` and can coincide;
    # `eq` *raises* on a procedure rather than answering, so a function type
    # must never be what makes a branch dead; and the variadic types classify
    # no values at all.
    return left in GROUNDS and right in GROUNDS and left != right


def disjoint_operands(
    left: CExpr,
    right: CExpr,
    context: Context,
    table: BindingTable,
) -> bool:
    # A `match` guard cannot bind the two ascriptions it has to compare, so the
    # pair is read here.
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
            # The mirrored bound is not a relation of its own: `k < x` is
            # `(not (or (lt x k) (eq x k)))`, and the rewrite recurses so only
            # `lt` and `eq` are ever read directly.
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
            # Operands of disjoint types never compare equal, so the test is
            # false whatever the target is: the consequent is reached by no
            # state at all, and the alternative is reached by every one of
            # them, which says nothing.
            #
            # This reads `eq` and not `lt`, because `eq` is total over data and
            # answers `#f` on a mismatch while `lt` raises — and a test that
            # raises is not a test that is false.
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
        case BooleanCExpr(value):
            return Support.integers() if value.value is polarity else Support.empty()
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


def contradictory(
    test: CExpr,
    consequent: bool,
    context: Context,
    table: BindingTable,
) -> bool:
    # Whether `test` cannot take the truth value `consequent`, which is to say
    # that the states giving it that value are none. The target is
    # `Symbol.toplevel()` because no reference resolves to it, so every reading
    # keyed on the target fails and only the facts that hold of every variable
    # are left to empty the support.
    return satisfying(test, Symbol.toplevel(), table, consequent, context).is_empty()


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
        support: Support = satisfying(guard.test, var, table, guard.consequent, context)
        rets.append(rets[-1].intersection(support))

    return rets


def unreachable(
    path: Path,
    program: Expr,
    table: BindingTable,
    context: Context,
) -> bool:
    # Whether no state reaches the end of `path`: some guard along it is false
    # in the polarity that reaching the end commits to. A malformed path raises
    # here as it does in `path_conditions`, but an unreadable guard does not —
    # it simply says nothing, and a path of such guards is reachable.
    guards: list[Guard] = path_conditions(path, program, table)

    return any(
        contradictory(guard.test, guard.consequent, context, table) for guard in guards
    )
