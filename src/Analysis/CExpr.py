from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Self, override

from Resolve import BindingTable, Unbound, resolve
from Syntax import Boolean, Char, Expr, Integer, Nil, SList, Symbol

__all__: list[str] = [
    "CExpr",
    "CExprError",
    "VarCExpr",
    "LitCExpr",
    "IntegerCExpr",
    "CharCExpr",
    "BooleanCExpr",
    "NilCExpr",
    "ListCExpr",
    "ConjunctionCExpr",
    "DisjunctionCExpr",
    "ComparisonCExpr",
    "NegationCExpr",
    "EquivalenceCExpr",
]


class CExprError(Exception):
    pass


# "CExpr" is the shortened name for "conditional expression".
class CExpr(ABC):
    @classmethod
    def from_expr(cls, expr: Expr, store: BindingTable | None = None) -> CExpr:
        if store is None:
            store = BindingTable()

        return convert(
            expr=expr,
            store=BindingTable(
                [*store, *map(Symbol, OPERATORS), *map(Symbol, UNARY_OPERATORS)]
            ),
        )

    @abstractmethod
    def to_expr(self: Self) -> Expr: ...


# 'VarCExpr(s)' corresponds to the S-expression 's'.
@dataclass()
class VarCExpr(CExpr):
    symbol: Symbol

    @override
    def to_expr(self: Self) -> Expr:
        return self.symbol


# The abstract base class for all 'CExpr' (conditional expression) literals.
class LitCExpr(CExpr):
    pass


# 'IntegerCExpr(n)' corresponds to a literal integer value.
@dataclass()
class IntegerCExpr(LitCExpr):
    value: Integer

    @override
    def to_expr(self: Self) -> Expr:
        return self.value


# 'CharCExpr(c)' corresponds to a literal character value.
@dataclass()
class CharCExpr(LitCExpr):
    value: Char

    @override
    def to_expr(self: Self) -> Expr:
        return self.value


# 'BooleanCExpr(b)' corresponds to a literal boolean value.
@dataclass()
class BooleanCExpr(LitCExpr):
    value: Boolean

    @override
    def to_expr(self: Self) -> Expr:
        return self.value


# 'NilCExpr()' corresponds to a literal nil value.
@dataclass()
class NilCExpr(LitCExpr):
    # Nil carries no payload, so the datum is reconstructed rather than
    # stored: every `Nil` compares equal to every other.
    @override
    def to_expr(self: Self) -> Expr:
        return Nil()


# 'AppCExpr(f, a, b, ...)' corresponds to the S-expression '(f a b ...)'.
@dataclass()
class ListCExpr(CExpr):
    items: list[Expr]

    @override
    def to_expr(self: Self) -> Expr:
        return SList(items=self.items)


# 'ConjunctionCExpr(a, b)' corresponds to the S-expression '(and a b)'.
@dataclass()
class ConjunctionCExpr(CExpr):
    left_operand: CExpr
    right_operand: CExpr

    @override
    def to_expr(self: Self):
        return SList(
            [
                Symbol("and"),
                self.left_operand.to_expr(),
                self.right_operand.to_expr(),
            ]
        )


# 'DisjunctionCExpr(a, b)' corresponds to the S-expression '(or a b)'.
@dataclass()
class DisjunctionCExpr(CExpr):
    left_operand: CExpr
    right_operand: CExpr

    @override
    def to_expr(self: Self):
        return SList(
            [
                Symbol("or"),
                self.left_operand.to_expr(),
                self.right_operand.to_expr(),
            ]
        )


# 'ComparisonCExpr(a, b)' corresponds to the S-expression '(lt a b)'.
@dataclass()
class ComparisonCExpr(CExpr):
    left_operand: CExpr
    right_operand: CExpr

    @override
    def to_expr(self: Self):
        return SList(
            [
                Symbol("lt"),
                self.left_operand.to_expr(),
                self.right_operand.to_expr(),
            ]
        )


# 'NegationCExpr(a)' corresponds to the S-expression '(not a)'.
@dataclass()
class NegationCExpr(CExpr):
    operand: CExpr

    @override
    def to_expr(self: Self):
        return SList(
            [
                Symbol("not"),
                self.operand.to_expr(),
            ]
        )


# 'EquivalenceCExpr(a, b)' corresponds to the S-expression '(eq a b)'.
@dataclass()
class EquivalenceCExpr(CExpr):
    left_operand: CExpr
    right_operand: CExpr

    @override
    def to_expr(self: Self):
        return SList(
            [
                Symbol("eq"),
                self.left_operand.to_expr(),
                self.right_operand.to_expr(),
            ]
        )


# The built-in operators that have a condition constructor of their own, each
# paired with the constructor its form builds and split by arity. A head that
# resolves to anything else spells an ordinary application.
OPERATORS: Final[dict[str, Callable[[CExpr, CExpr], CExpr]]] = {
    "and": ConjunctionCExpr,
    "or": DisjunctionCExpr,
    "lt": ComparisonCExpr,
    "eq": EquivalenceCExpr,
}

UNARY_OPERATORS: Final[dict[str, Callable[[CExpr], CExpr]]] = {
    "not": NegationCExpr,
}


def constructor[T](
    head: Symbol, store: BindingTable, operators: dict[str, T]
) -> T | None:
    # Which condition constructor a head names, or `None` when the head is an
    # ordinary operator. Resolution decides this and spelling does not: a
    # program with a binder of its own for `and` has an `and` that is not the
    # built-in one, and the condition must read it as the application it is.
    try:
        binder: Symbol = resolve(head, store).binder

        # Every built-in binder is bare, so a binder carrying scopes is one the
        # program introduced that merely shares a built-in's spelling.
        return None if binder.scopes else operators.get(binder.identifier)
    except Unbound:
        # Not a failure here: every head that is neither a built-in nor
        # program-bound arrives here, and all of them are applications.
        return None


def convert(expr: Expr, store: BindingTable) -> CExpr:
    match expr:
        case Symbol() as symbol:
            return VarCExpr(symbol)
        case Integer() as value:
            return IntegerCExpr(value)
        case Char() as value:
            return CharCExpr(value)
        case Boolean() as value:
            return BooleanCExpr(value)
        case Nil():
            return NilCExpr()
        case SList([Symbol() as head, operand]):
            # A two-item list can only be the unary built-in: `(not a b)` is an
            # application of `not`, whatever `not` resolves to.
            build_unary: Callable[[CExpr], CExpr] | None = constructor(
                head, store, UNARY_OPERATORS
            )

            if build_unary is None:
                return ListCExpr([head, operand])
            else:
                return build_unary(convert(operand, store))
        case SList([Symbol() as head, left, right]):
            # The binary built-in forms need exactly a three-item list:
            # `(and a b c)` is an application of `and`, whatever `and`
            # resolves to.
            build: Callable[[CExpr, CExpr], CExpr] | None = constructor(
                head, store, OPERATORS
            )

            if build is None:
                return ListCExpr([head, left, right])
            else:
                return build(convert(left, store), convert(right, store))
        case SList(items):
            # The empty combination is not refused: `parse` has already done
            # that, and this conversion is total on the data it represents.
            return ListCExpr(items)
        case _:
            # Any sort the condition language does not represent, `Keyword`
            # being the first: a keyword names an argument position and states
            # no condition. The arm is also what a checker needs, since
            # exhaustiveness over an ABC is not something it can see, and
            # falling through would return `None` from a function annotated
            # `CExpr`.
            raise CExprError(f"not a convertible expression: {expr}")
