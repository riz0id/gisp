from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Self, override

from Resolve import BindingTable, Unbound, resolve
from Syntax import Char, Expr, Integer, SList, Symbol, binder_name

__all__: list[str] = [
    "CExpr",
    "CExprError",
    "VarCExpr",
    "LitCExpr",
    "IntegerCExpr",
    "CharCExpr",
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
        case SList([Symbol() as head, operand]) if (head.identifier == "forall") and (
            bound := binder_name(operand)
        ) is not None:
            return VarCExpr(bound)
        case SList([Symbol() as head, operand]):
            build_unary: Callable[[CExpr], CExpr] | None = constructor(
                head, store, UNARY_OPERATORS
            )

            if build_unary is None:
                return ListCExpr([head, operand])
            else:
                return build_unary(convert(operand, store))
        case SList([Symbol() as head, left, right]):
            build: Callable[[CExpr, CExpr], CExpr] | None = constructor(
                head, store, OPERATORS
            )

            if build is None:
                return ListCExpr([head, left, right])
            else:
                return build(convert(left, store), convert(right, store))
        case SList(items):
            return ListCExpr(items)
        case _:
            raise CExprError(f"not a convertible expression: {expr}")
